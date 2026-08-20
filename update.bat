@echo off
setlocal
set "CURRENT_DIR=%CD%"
echo ***** Updating MoneyPrinter Custom *****

rem 拉取最新代码（本地有未提交修改时 git pull 会失败并给出提示）。
git -C "%CURRENT_DIR%" pull --rebase
if errorlevel 1 (
    echo ***** Git pull failed. Commit or stash local changes first. *****
    pause
    exit /b 1
)

rem 更新依赖。优先使用项目自带的 .venv 与 uv，避免破坏其它项目。
if exist "%CURRENT_DIR%\.venv\Scripts\python.exe" (
    echo ***** Refreshing project virtual environment *****
    "%CURRENT_DIR%\.venv\Scripts\python.exe" -m pip install -r requirements.txt
) else if exist "%CURRENT_DIR%\lib\python\python.exe" (
    echo ***** Refreshing embedded Python environment *****
    "%CURRENT_DIR%\lib\python\python.exe" -m pip install -r requirements.txt
) else (
    where uv >nul 2>nul
    if not errorlevel 1 (
        echo ***** Refreshing dependencies with uv *****
        uv sync --frozen
    ) else (
        echo ***** Neither project Python nor uv found. Skipping dependency update. *****
        echo ***** Install Python 3.11+, then run: pip install -r requirements.txt *****
    )
)

echo ***** Update finished. Launch with webui.bat *****
pause