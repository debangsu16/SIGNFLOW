@echo off
setlocal

cd "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.10 virtual environment...
    py -3.10 -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -c "import cv2, mediapipe, tensorflow, PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

echo Starting Sign Language Detector...
".venv\Scripts\python.exe" desktop_app.py
if errorlevel 1 goto :error
goto :end

:error
echo.
echo Could not start the application.
echo Make sure Python 3.10 is installed and available through the py launcher.
pause
exit /b 1

:end
endlocal
