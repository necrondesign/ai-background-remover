@echo off
REM ---------------------------------------------------------------------------
REM build_windows.bat — Build AI Background Remover для Windows
REM
REM Использование:
REM   1. Открыть CMD в корне проекта
REM   2. python -m venv .venv
REM   3. .venv\Scripts\activate
REM   4. pip install -r requirements.txt
REM   5. pip install pyinstaller
REM   6. scripts\build_windows.bat
REM ---------------------------------------------------------------------------

set APP_NAME=AI Background Remover
set APP_VERSION=1.1

echo === Building %APP_NAME% v%APP_VERSION% for Windows ===

REM Проверяем venv
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Create it first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

REM Скачиваем модель если нет
set HF_CACHE=%USERPROFILE%\.cache\huggingface\hub\models--ZhengPeng7--BiRefNet
if not exist "%HF_CACHE%" (
    echo Downloading BiRefNet model...
    .venv\Scripts\python.exe -c "from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True); print('Model downloaded.')"
)

REM Собираем через PyInstaller
echo Building with PyInstaller...
.venv\Scripts\pyinstaller.exe --noconfirm rmbg_remover.spec

if %ERRORLEVEL% neq 0 (
    echo ERROR: PyInstaller build failed!
    exit /b 1
)

echo.
echo === Build complete ===
echo Output: dist\AI Background Remover\
echo.
echo To create installer, use Inno Setup with scripts\installer.iss
echo Or just zip the "dist\AI Background Remover" folder.
