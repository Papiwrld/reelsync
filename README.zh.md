<div align="center">

# ReelSync 💸

### AI 短视频生成工具

English | [简体中文](README.md)

</div>

ReelSync 是一款根据主题或脚本自动生成短视频的工具。输入主题后，它会自动撰写脚本、匹配素材、生成字幕与背景音乐，并合成高清视频。

本项目 fork 自 [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)（原作者 [harry0703](https://github.com/harry0703)），在其扎实的基础上增加了我们需要的功能，形成了自己的版本。完整致谢见文末。

---

## 功能概览

**素材来源**

- 多供应商支持：Pexels、Pixabay、Coverr、自定义媒体 API、网页抓取与 Pollinations。
- **免费模式（Free Mode）**：未配置任何图库 Key、自定义 API 或网页抓取时，自动启用免费的 Pollinations 作为素材供应商——配音使用免费的 Edge TTS，整条流水线零付费 API Key 即可跑通。
- **Auto（综合所有素材源择优）**：并发查询所有已配置供应商；未配置（如缺 Key）的供应商自动跳过，只需一个可用 Key 即可。
- Auto 模式下可通过 WebUI 的“Auto 素材来源”多选框或配置 `auto_providers` 限定参与来源，顺序固定为内置优先级。
- 自定义 API 支持多种响应格式（`standard`、`openai`、`url_list`）。
- 可选的网页抓取通道（基于 yt-dlp），用于获取图库之外的素材。
- 线程池并发下载，并按分辨率、时长与来源优先级智能排序选片。

**脚本与配音**

- AI 生成脚本（支持多种 LLM 供应商），或直接粘贴自备脚本。
- 语音合成：默认使用免费的 Edge TTS，另支持 Azure、SiliconFlow、Gemini、ElevenLabs 等。

**字幕**

- 根据配音时间轴自动生成。
- 2~4 词自然短语分句，避免单字闪烁。
- 可配置字体、字号、颜色、描边、背景、位置与文字大小写。

**视频合成**

- 多种转场模式，包括 **"Mix" 混合转场**（跨片段交叉溶解）。
- 静态图片的 Ken Burns 推拉摇移效果。
- 支持竖屏（9:16）与横屏（16:9）。

**音频**

- 背景音乐，音量可调。
- **Smart Audio Ducking 智能闪避**：语音播放时自动压低背景音乐。
- 可选氛围铺底与转场音效（来自 `resource/sfx/`）。

**智能内容策划**

- **智能内容策划（Agentic Planning）**——可选的策略图谱（主题分析 → 内容策略 → 钩子选择 → 叙事规划 → 文案 → 评审修订），替代单一通用提示词，由内容画像引导。
- **内容画像（Content Profile）**——可复用的创作者人设，设定语气、节奏、钩子风格、素材与字幕规则，可在 WebUI 中自行编写。
- **内容智能（Content Intelligence）**——一次性定义赛道、受众、平台、形式与目标；管线据此自适应研究深度、事实核查、叙事、标题与视觉（手动 → 自动驾驶各级别）。
- **导入主页（Import Profile）**——粘贴 TikTok / Instagram / Facebook / X / YouTube 主页链接，赛道、受众与语气字段自动回填（公开搜索摘要，以及可行的公开页面元数据 yt-dlp）；链接同时加入研究笔记。来源诚实标注：“公开数据”或“模型知识”。
- **发现选题（Discover Topics）**——为你的赛道生成带评分的选题候选，一键用作视频主题。
- **研究编排（Research Orchestrator）**——风险自适应的来源发现、论点提取与事实核查；来源来自模型知识、你的笔记或可选的网页搜索供应商（`[research]`），绝不伪造；无法确认的论点必须在文案中明确限定。

**WebUI**

- 深色主题四栏工作台：脚本与关键词、视频设置、音频与配音、字幕与输出。
- 顶栏内置用户手册（📖），覆盖完整工作流程与每个功能的用法。
- 任务管理器、生成进度日志与结果预览；可重播或重新生成任意历史任务。
- 多语言界面（英文、中文及其它 7 种语言）。

---

## 环境要求

- Python 3.11+
- 系统 PATH 中包含 FFmpeg

## 快速开始

ReelSync 以源码方式运行，不提供 `pip install reelsync` 安装包。`pip install -r requirements.txt` 仅安装依赖；配置文件、`resource/` 资源与生成结果都保留在项目目录内。

```bash
git clone https://github.com/Papiwrld/reelsync.git
cd reelsync

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

cp config.example.toml config.toml
```

编辑 `config.toml`，至少配置一个视频素材供应商（如 Pexels）；如需 AI 写脚本，再配置 LLM 供应商。配音默认使用免费的 Edge TTS，无需 API Key。

启动 WebUI：

```bash
streamlit run webui/Main.py
```

浏览器访问 http://127.0.0.1:8501。

也支持命令行：

```bash
python cli.py --help
```

## 配置说明

所有配置都在 `config.toml` 中（完整选项与注释见 `config.example.toml`）。

| 配置段 | 作用 |
|---|---|
| `[app]` | 监听地址与端口、项目名、日志级别 |
| `[media]` | 自定义媒体 API 地址、密钥、请求方式、响应格式、混合回退 |
| `[pexels]` / `[pixabay]` / `[coverr]` | 图库素材供应商 API Key |
| `[llm_providers]` | AI 写稿与关键词提取的供应商 |
| `[azure]` / `[siliconflow]` / `[elevenlabs]` | 可选的语音合成供应商 |
| `[whisper]` | 字幕识别模型 |
| `[ui]` | WebUI 默认值（语言、字幕设置等） |
| `[agentic]` | 智能策划（文案评审最大修订轮数） |
| `[research]` | 研究供应商：可选通用网页搜索端点、API Key、缓存 TTL |

### 自定义媒体 API

任何返回视频 URL 列表的接口都可以接入：

```toml
[media]
custom_api_url = "https://your-provider.com/api/search"
custom_api_key = "your-key"
# custom_api_method = "POST"            # "POST"（默认）或 "GET"
# custom_api_response_format = "standard"  # "standard"、"openai" 或 "url_list"
# custom_api_extra_headers = ""         # 可选：附加请求头（JSON 对象）
# custom_api_extra_body = ""            # 可选：合并进请求体的 JSON 对象
# hybrid_video_mode = true              # 自定义 API 失败时回退到 Pexels
```

响应字段格式参考 `app/services/custom_media.py`。

### 网页抓取与 Pollinations（免费模式）

网页抓取（内置 yt-dlp）可为 Auto 模式补充图库之外的 CC0/CC-BY 素材：

```toml
[app]
enable_web_scraping = true
```

Pollinations 是免费无密钥的图片素材供应商。当其它供应商都未配置时会自动启用（免费模式），也可以显式强制开关：

```toml
[app]
# enable_pollinations = true    # 强制开启（或 false 强制关闭）
# pollinations_image_model = "flux"
```

限制 Auto 模式参与来源：

```toml
[app]
# 只有列表中的来源参与；顺序始终为内置优先级。
# auto_providers = ["custom_api", "pexels", "pixabay", "coverr", "web_scrape", "pollinations"]
```

### 研究与分析（可选）

研究层绝不伪造数据：未配置供应商时，来源仅为模型知识与你的笔记（会明确标注）。要获取真实的最新来源，请将 `[research]` 指向一个通用 JSON 搜索端点——只获取搜索元数据（不抓取页面内容），研究层保持 SSRF 安全：

```toml
[research]
provider = "web_search"
base_url = "https://your-search-endpoint"   # 返回 {"results": [{"title", "url", "snippet"}]}
# api_key = ""
# ttl_hours = 24
```

### 自定义音效

将音频文件放入以下目录即可启用音效与氛围层：

```
resource/sfx/
├── transitions/    # 片段转场时播放
└── atmosphere/     # 低频氛围层循环混入
```

支持 `.mp3`、`.wav`、`.ogg`。目录不存在时自动跳过，不会报错。

---

## 致谢与许可证

ReelSync fork 自 **MoneyPrinterTurbo**，原作者为 [harry0703](https://github.com/harry0703/MoneyPrinterTurbo)。底层的脚本生成、视频拼接、TTS 集成与任务系统等核心管线来自该项目，衷心感谢原作者与所有贡献者。

两个项目均以 **MIT License** 开源，详见 [LICENSE](LICENSE)。

## 打包说明

ReelSync 以源码形式分发，不构建安装包。`pyproject.toml` 仅用于依赖锁定（`uv sync` / `pip install -r requirements.txt`），wheel 构建目标已按设计关闭——应用运行时依赖项目目录结构（`config.toml`、`resource/`、`storage/`）。

## 说明

- 视频在本地渲染，不会上传到任何云端服务。
- 首次使用字幕识别时会从 Hugging Face 下载 Whisper 模型；若下载失败（部分地区常见），可手动下载 `whisper-large-v3` 并放置到 `models/whisper-large-v3/`。
