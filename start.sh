#!/bin/bash

echo "========================================"
echo "  Flappy Bird Deep Q-Learning"
echo "  Inicio Rápido (Sin Docker)"
echo "========================================"
echo ""

echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python no está instalado"
    exit 1
fi

python3 --version

echo ""
echo "Instalando dependencias..."
pip3 install -q torch fastapi uvicorn websockets numpy python-multipart

echo ""
echo "Iniciando servidor backend..."
cd "$(dirname "$0")"
python3 backend/api_server.py &
BACKEND_PID=$!

sleep 3

echo ""
echo "Iniciando servidor frontend..."
python3 -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

sleep 2

echo ""
echo "========================================"
echo "  Sistema iniciado correctamente!"
echo "========================================"
echo ""
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo ""
echo "Presiona Ctrl+C para detener el sistema..."

# Trap Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Esperar
wait
