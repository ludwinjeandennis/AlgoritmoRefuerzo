"""
Servidor FastAPI con WebSocket para comunicación en tiempo real
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import Optional, Set
import uvicorn
from environment import FlappyBirdEnv
from agent import DQNAgent
from training import Trainer


app = FastAPI(title="Flappy Bird RL API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global
trainer: Optional[Trainer] = None
env: Optional[FlappyBirdEnv] = None
agent: Optional[DQNAgent] = None
is_training = False
active_connections: Set[WebSocket] = set()
current_model_name = "Nuevo Modelo"  # Nombre del modelo actual


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Flappy Bird RL API"}


@app.get("/models")
async def list_models():
    """Lista todos los modelos disponibles en la carpeta models/."""
    import os
    import glob
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return {"models": []}
    
    # Buscar todos los archivos .pth
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    
    # Obtener información de cada modelo
    models_info = []
    for filepath in model_files:
        filename = os.path.basename(filepath)
        size = os.path.getsize(filepath)
        mtime = os.path.getmtime(filepath)
        
        # Normalizar ruta para que funcione en Windows y navegador
        # Usar solo el nombre del archivo, el backend agregará "models/" al cargar
        models_info.append({
            "filename": filename,
            "filepath": filename,  # Solo el nombre del archivo
            "size_kb": round(size / 1024, 2),
            "modified": mtime
        })
    
    # Ordenar por fecha de modificación (más reciente primero)
    models_info.sort(key=lambda x: x['modified'], reverse=True)
    
    return {"models": models_info}


async def run_training_with_visualization():
    """Ejecuta el entrenamiento con visualización en tiempo real."""
    global trainer, env, agent, is_training
    
    if trainer is None:
        trainer = Trainer(episodes=10000, fast_mode=False, train_frequency=1)
        env = trainer.env
        agent = trainer.agent
    
    # Continuar desde el episodio actual (no reiniciar)
    start_episode = trainer.current_episode + 1 if trainer.current_episode > 0 else 1
    
    print(f"{'Continuando' if start_episode > 1 else 'Iniciando'} entrenamiento desde episodio {start_episode}...")
    
    for episode in range(start_episode, trainer.episodes + 1):
        if not is_training:
            break
            
        trainer.current_episode = episode
        
        # Entrenar episodio con visualización frame por frame
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        while not env.done and is_training:
            # Seleccionar acción
            action = agent.select_action(state)
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Almacenar experiencia
            agent.store_experience(state, action, reward, next_state, done)
            
            # Entrenar
            if env.frames % trainer.train_frequency == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
            
            state = next_state
            
            # Pequeña pausa para permitir visualización
            await asyncio.sleep(0.001)
        
        # Calcular métricas del episodio
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        score = info.get('score', 0)
        
        # Actualizar best score
        if score > trainer.best_score:
            trainer.best_score = score
            trainer.save_best_model()
        
        # Guardar métricas
        trainer.episode_scores.append(score)
        trainer.episode_losses.append(avg_loss)
        trainer.episode_epsilons.append(agent.epsilon)
        
        # Log cada 10 episodios
        if episode % 10 == 0:
            avg_score = sum(trainer.episode_scores[-10:]) / 10
            print(f"Episodio {episode}/{trainer.episodes} | "
                  f"Score: {score} | "
                  f"Avg Score (10): {avg_score:.1f} | "
                  f"Best: {trainer.best_score} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Guardar checkpoint
        if episode % trainer.save_frequency == 0:
            trainer.save_checkpoint(episode)
            trainer.save_training_log()
    
    is_training = False
    print("Entrenamiento completado o pausado!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para streaming de datos del juego en tiempo real.
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    global trainer, env, agent, is_training, current_model_name
    
    # Inicializar si es necesario
    if trainer is None:
        trainer = Trainer(fast_mode=False)
        env = trainer.env
        agent = trainer.agent
    
    try:
        while True:
            # Recibir comandos del cliente
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                command = json.loads(data)
                
                if command.get('action') == 'start':
                    if not is_training:
                        is_training = True
                        asyncio.create_task(run_training_with_visualization())
                        
                elif command.get('action') == 'pause':
                    is_training = False
                    
                elif command.get('action') == 'save':
                    if trainer:
                        episode = trainer.current_episode
                        filename = f"checkpoint_episode{episode}.pth"
                        trainer.save_checkpoint(episode)
                        current_model_name = filename
                        print(f"Modelo guardado: {filename}")
                        
                elif command.get('action') == 'load':
                    if trainer:
                        import os
                        filename = command.get('filepath', 'best_model.pth')
                        
                        # Construir ruta completa (el frontend solo envía el nombre del archivo)
                        if not filename.startswith('models'):
                            filepath = os.path.join('models', filename)
                        else:
                            filepath = filename
                        
                        # Verificar si el archivo existe
                        if os.path.exists(filepath):
                            try:
                                trainer.load_model(filepath)
                                env = trainer.env
                                agent = trainer.agent
                                current_model_name = os.path.basename(filepath)
                                print(f"Modelo cargado: {current_model_name}")
                                print(f"  - Episodio: {trainer.current_episode}")
                                print(f"  - Best Score: {trainer.best_score}")
                                print(f"  - Epsilon: {agent.epsilon:.3f}")
                            except Exception as e:
                                print(f"Error al cargar modelo: {e}")
                                current_model_name = "Error al cargar"
                        else:
                            print(f"Archivo no encontrado: {filepath}")
                            current_model_name = "Archivo no encontrado"
                        
            except asyncio.TimeoutError:
                pass
            
            # Enviar estado del juego
            if env is not None and agent is not None and trainer is not None:
                game_state = env.render()
                
                # Agregar métricas
                metrics = {
                    'episode': trainer.current_episode,
                    'score': env.score,
                    'best_score': trainer.best_score,
                    'epsilon': agent.epsilon,
                    'loss': agent.last_loss,
                    'avg_q_value': agent.avg_q_value,
                    'is_training': is_training,
                    'model_name': current_model_name  # Nombre del modelo actual
                }
                
                # Datos completos
                response = {
                    'game': game_state,
                    'metrics': metrics,
                    'history': {
                        'scores': trainer.episode_scores[-100:] if trainer.episode_scores else [],
                        'losses': trainer.episode_losses[-100:] if trainer.episode_losses else [],
                        'epsilons': trainer.episode_epsilons[-100:] if trainer.episode_epsilons else []
                    }
                }
                
                await websocket.send_json(response)
            
            # Control de FPS (60 FPS para visualización)
            await asyncio.sleep(1/60)
            
    except WebSocketDisconnect:
        print("Cliente desconectado")
        active_connections.discard(websocket)
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        active_connections.discard(websocket)


if __name__ == "__main__":
    print("Iniciando servidor FastAPI...")
    print("API disponible en: http://localhost:8000")
    print("WebSocket disponible en: ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)
