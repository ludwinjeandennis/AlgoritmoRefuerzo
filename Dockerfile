FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Crear directorios para persistencia
RUN mkdir -p /app/models /app/logs

# Exponer puertos
EXPOSE 8000 3000

# Comando por defecto
CMD ["sh", "-c", "python backend/api_server.py & python -m http.server 3000 --directory frontend"]
