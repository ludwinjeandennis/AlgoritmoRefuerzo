@echo off
echo ========================================
echo   Flappy Bird Deep Q-Learning
echo   Inicio Rapido (Sin Docker)
echo ========================================
echo.

echo Verificando Python...
python --version
if errorlevel 1 (
    echo ERROR: Python no esta instalado
    pause
    exit /b 1
)

echo.
echo Instalando dependencias...
pip install -q torch fastapi uvicorn websockets numpy python-multipart
if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

echo.
echo Iniciando servidor backend...
start "Flappy Bird Backend" cmd /k "cd /d %~dp0 && python backend/api_server.py"

timeout /t 3 /nobreak > nul

echo.
echo Iniciando servidor frontend...
start "Flappy Bird Frontend" cmd /k "cd /d %~dp0 && python -m http.server 3000 --directory frontend"

timeout /t 2 /nobreak > nul

echo.
echo ========================================
echo   Sistema iniciado correctamente!
echo ========================================
echo.
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo.
echo Presiona cualquier tecla para abrir el navegador...
pause > nul

start http://localhost:3000

echo.
echo Para detener el sistema, cierra las ventanas del backend y frontend.
echo.
pause
