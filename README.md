# 🐦 Flappy Bird Deep Q-Learning

Sistema completo de aprendizaje por refuerzo (Deep Q-Network) para entrenar un agente autónomo que juega Flappy Bird, con visualización en tiempo real y ejecución local mediante Docker.

## 🎯 Características

- **Agente DQN**: Red neuronal profunda con PyTorch (3 → 64 → 64 → 2)
- **Replay Memory**: Buffer de 50,000 experiencias para entrenamiento estable
- **Epsilon-Greedy**: Exploración adaptativa con decay automático
- **Target Network**: Red objetivo para estabilidad en el aprendizaje
- **Visualización en Tiempo Real**: Canvas HTML5 con animaciones pixel-art
- **Gráficas Interactivas**: Métricas en vivo con Chart.js
- **WebSocket**: Comunicación bidireccional entre backend y frontend
- **Persistencia**: Sistema de guardado/carga de modelos
- **Docker**: Ejecución completa con un solo comando

## 🏗️ Arquitectura

```
flappy-bird-rl/
├── backend/
│   ├── neural_network.py    # Red DQN con PyTorch
│   ├── agent.py              # Agente Q-Learning
│   ├── environment.py        # Simulación del juego
│   ├── training.py           # Loop de entrenamiento
│   └── api_server.py         # Servidor FastAPI WebSocket
├── frontend/
│   ├── index.html            # Estructura HTML
│   ├── style.css             # Estilos pixel-art
│   ├── game.js               # Renderizado y WebSocket
│   └── assets/               # Sprites (generados en canvas)
├── models/                   # Checkpoints guardados
├── logs/                     # Logs de entrenamiento
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Instalación y Uso

### Opción 1: Con Docker (Recomendado)

1. **Construir la imagen**:
```bash
docker-compose build
```

2. **Iniciar el sistema**:
```bash
docker-compose up
```

3. **Acceder a la interfaz**:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000

### Opción 2: Sin Docker

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

2. **Iniciar el servidor backend**:
```bash
python backend/api_server.py
```

3. **Iniciar el servidor frontend** (en otra terminal):
```bash
python -m http.server 3000 --directory frontend
```

4. **Acceder a la interfaz**:
   - Frontend: http://localhost:3000

## 🎮 Uso de la Interfaz

### Panel de Control

- **▶ Iniciar**: Comienza el entrenamiento del agente
- **⏸ Pausar**: Pausa el entrenamiento actual
- **💾 Guardar**: Guarda el modelo actual
- **📂 Cargar**: Carga el mejor modelo guardado

### Estadísticas en Tiempo Real

- **Episodio**: Número de episodio actual
- **Score**: Puntuación del episodio actual
- **Best Score**: Mejor puntuación alcanzada
- **Epsilon**: Tasa de exploración actual
- **Loss**: Pérdida de la red neuronal
- **Q-Value**: Valor Q promedio

### Gráficas

1. **Score por Episodio**: Progreso del aprendizaje (últimos 100 episodios)
2. **Pérdida de la Red**: Convergencia del entrenamiento
3. **Epsilon Decay**: Reducción de la exploración

## 🧠 Detalles Técnicos

### Estado del Agente
```python
[distancia_horizontal_tubería, distancia_vertical_tubería, velocidad_pajaro]
```

### Acciones
- `0`: No hacer nada
- `1`: Saltar

### Recompensas
- `+1`: Por cada frame sobreviviendo
- `+10`: Por pasar una tubería
- `-1000`: Por colisión

### Hiperparámetros
- **Learning Rate**: 0.0001
- **Gamma (descuento)**: 0.99
- **Epsilon inicial**: 1.0
- **Epsilon final**: 0.01
- **Epsilon decay**: 0.9995
- **Batch size**: 32
- **Replay memory**: 50,000 experiencias
- **Target update**: cada 1000 pasos

### Arquitectura de la Red
```
Input Layer:    3 neuronas (estado)
Hidden Layer 1: 64 neuronas (ReLU)
Hidden Layer 2: 64 neuronas (ReLU)
Output Layer:   2 neuronas (Q-values)
```

## 📊 Persistencia

### Modelos Guardados
Los modelos se guardan automáticamente en `models/`:
- `checkpoint_episode100.pth`, `checkpoint_episode200.pth`, etc.
- `best_model.pth`: Mejor modelo por score

### Logs de Entrenamiento
Los logs se guardan en `logs/training_log.json`:
```json
{
  "episodes": [
    {
      "episode": 1,
      "score": 5,
      "epsilon": 0.995,
      "loss": 0.234
    }
  ],
  "best_score": 50
}
```

## 🎨 Diseño Visual

El frontend replica fielmente la estética retro de Flappy Bird:
- **Paleta de colores original**: Cielo azul (#70c5ce), pájaro amarillo (#ffcc00), tuberías verdes (#5ac54f)
- **Fuente pixel-art**: Press Start 2P de Google Fonts
- **Animaciones suaves**: Sistema de partículas, nubes en movimiento, suelo animado
- **Renderizado pixelado**: Canvas con `image-rendering: pixelated`

## 🔧 Configuración Avanzada

### Modo Fast (Entrenamiento Rápido)
Para entrenar sin visualización a máxima velocidad:
```python
trainer = Trainer(episodes=10000, fast_mode=True)
trainer.train()
```

### Cargar Modelo Preentrenado
```python
trainer.load_model('models/best_model.pth')
```

### Ajustar Hiperparámetros
Edita los valores en `backend/agent.py`:
```python
agent = DQNAgent(
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_decay=0.9995
)
```

## 📝 Troubleshooting

### El WebSocket no conecta
- Verifica que el backend esté corriendo en el puerto 8000
- Revisa la consola del navegador para errores
- Asegúrate de que no haya firewalls bloqueando el puerto

### El entrenamiento no mejora
- Aumenta el número de episodios
- Ajusta el learning rate (prueba 0.001 o 0.00001)
- Verifica que epsilon esté decayendo correctamente

### Docker no inicia
- Asegúrate de tener Docker instalado y corriendo
- Verifica que los puertos 8000 y 3000 estén disponibles
- Revisa los logs con `docker-compose logs`

## 🎯 Resultados Esperados

Con el entrenamiento adecuado (1000+ episodios), el agente debería:
- Alcanzar scores consistentes de 50+
- Reducir epsilon a ~0.01
- Mostrar convergencia en la pérdida
- Navegar exitosamente entre tuberías

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

**¡Disfruta entrenando tu agente de Flappy Bird! 🚀**
