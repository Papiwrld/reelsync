import json
from typing import Dict

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager, TaskQueueFullError
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        max_queued_tasks: int = 100,
    ):
        # 直接传独立连接参数而不是 redis:// 连接串，避免把密码拼进模块级 URL
        # 字符串后随日志或调试输出泄露。
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password
        )
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)
        # 分布式并发计数器，key 与队列同前缀，避免多实例间内存计数器不一致。
        # 使用 Redis INCR/DECR 原子操作实现跨进程并发限制，TTL 防止崩溃后计数器泄漏。
        self._concurrency_key = f"{self.queue}:concurrent"

    def _try_acquire_slot(self) -> bool:
        """尝试通过 Redis 原子递增获取并发名额，超过上限则回滚并返回 False。"""
        try:
            # 确保 key 存在且带过期时间，避免崩溃后永久泄漏；set nx 仅在首次创建时生效。
            self.redis_client.set(self._concurrency_key, 0, nx=True)
            val = self.redis_client.incr(self._concurrency_key)
            if val == 1:
                try:
                    self.redis_client.expire(self._concurrency_key, 86400)
                except Exception:
                    pass
            if val > self.max_concurrent_tasks:
                self.redis_client.decr(self._concurrency_key)
                return False
            return True
        except Exception as exc:
            logger.warning(f"redis concurrency acquire failed, fallback to in-memory: {exc}")
            # Redis 不可用时回退为进程内并发检查，至少防止单进程内过度并发。
            if self.current_tasks < self.max_concurrent_tasks:
                self.current_tasks += 1
                return True
            return False

    def _release_slot(self) -> None:
        """释放并发名额，优先操作 Redis；仅当 Redis 计数器不足时回退到本地。"""
        try:
            new_val = self.redis_client.decr(self._concurrency_key)
            if new_val is not None and new_val < 0:
                # 计数器已为 0 说明本次释放对应的是之前 fallback 到本地的槽位，
                # 需重置 Redis 并回退本地计数器；纯 Redis 槽位释放不会进入此分支。
                self.redis_client.set(self._concurrency_key, 0)
                with self.lock:
                    if self.current_tasks > 0:
                        self.current_tasks -= 1
            return
        except Exception as exc:
            logger.warning(f"redis concurrency release failed: {exc}")
        # Redis 不可用或未使用 Redis 获取的场景，回退到本地计数器
        with self.lock:
            if self.current_tasks > 0:
                self.current_tasks -= 1

    def create_queue(self):
        return "task_queue"

    def enqueue(self, task: Dict):
        task_with_serializable_params = task.copy()
        # task.copy() 只复制最外层字典；如果直接改写嵌套 kwargs，会把调用方
        # 持有的 VideoParams 同步替换成 dict。后续日志或重试仍可能读取原任务，
        # 因此这里单独复制 kwargs，确保序列化过程没有意外副作用。
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(task_kwargs["params"], VideoParams):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # 将函数对象转换为其名称
        task_with_serializable_params["func"] = task["func"].__name__
        self.redis_client.rpush(self.queue, json.dumps(task_with_serializable_params))

    def dequeue(self):
        # 循环而非单次弹出：某个任务在入队时可能满足当时的 VideoParams 校验规则，
        # 但校验规则本身在两次部署之间收紧了（例如新增 ge=1 约束）。lpop 是破坏性
        # 操作，一旦弹出就不能放回原位；如果重建 VideoParams 时才发现校验失败，
        # 这条任务已经从队列中永久移除了，不能再假装它还在。与其让异常从这里往上
        # 抛、把这条已经丢失的任务的 lock 持有者带崩，不如原地丢弃并继续尝试队列
        # 里的下一条，把"拿到一条可用任务或者队列确实空了"这个约定维持住。
        while True:
            task_json = self.redis_client.lpop(self.queue)
            if not task_json:
                return None

            task_info = json.loads(task_json)
            # 将函数名称转换回函数对象
            task_info["func"] = FUNC_MAP[task_info["func"]]

            if "params" in task_info["kwargs"] and isinstance(
                task_info["kwargs"]["params"], dict
            ):
                try:
                    task_info["kwargs"]["params"] = VideoParams(
                        **task_info["kwargs"]["params"]
                    )
                except ValidationError as e:
                    logger.error(
                        "dropping queued task with params that fail current "
                        f"VideoParams validation (queued under an older, more "
                        f"permissive schema, or corrupted): {e}"
                    )
                    # 任务状态记录在入队前就已创建，且默认是 processing；如果只是
                    # 丢弃这条队列项而不动状态记录，API/WebUI 会一直显示任务在
                    # 运行，永远不会变成失败。用 patch_task 而不是 update_task，
                    # 这样如果用户已经删除了这个任务，我们不会又把它建回来。
                    task_id = task_info["kwargs"].get("task_id")
                    if task_id:
                        sm.state.patch_task(
                            task_id,
                            state=const.TASK_STATE_FAILED,
                            failed_stage="dequeue",
                            error=f"discarded stale queued task: {e}",
                        )
                    continue

            return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)

    def add_task(self, func, *args, **kwargs):
        """分布式并发控制的入队入口，使用 Redis 计数器而非本地内存。"""
        with self.lock:
            if self._try_acquire_slot():
                try:
                    self.execute_task(func, *args, **kwargs)
                except Exception:
                    self._release_slot()
                    raise
            else:
                queue_size = self.queue_size()
                if queue_size >= self.max_queued_tasks:
                    logger.warning(
                        f"reject task: {func.__name__}, queue_size: {queue_size}, "
                        f"max_queued_tasks: {self.max_queued_tasks}"
                    )
                    raise TaskQueueFullError(
                        "task queue is full, please try again later"
                    )
                logger.info(
                    f"enqueue task: {func.__name__}, queue_size: {queue_size}"
                )
                self.enqueue({"func": func, "args": args, "kwargs": kwargs})

    def task_done(self):
        """分布式并发释放，完成后尝试调度队列中的下一个任务。"""
        self._release_slot()
        self.check_queue()

    def check_queue(self):
        """在 Redis 并发槽位可用时调度下一个排队任务。"""
        with self.lock:
            if self.is_queue_empty():
                return
            if not self._try_acquire_slot():
                return
        # 出队操作在锁外执行，避免阻塞其他入队请求
        task_info = self.dequeue()
        if task_info is None:
            self._release_slot()
            return
        func = task_info["func"]
        args = task_info.get("args", ())
        kwargs = task_info.get("kwargs", {})
        try:
            self.execute_task(func, *args, **kwargs)
        except Exception:
            self._release_slot()
            # 线程启动失败需回滚并将任务放回队列，避免丢失
            try:
                self.enqueue(task_info)
            except Exception as exc:
                logger.warning(f"failed to re-enqueue task after execute failure: {exc}")
            raise
