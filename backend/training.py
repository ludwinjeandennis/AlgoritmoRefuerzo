"""
Loop principal de entrenamiento del agente DQN
"""
import json
import os
from datetime import datetime
from typing import Dict, List
from environment import FlappyBirdEnv
from agent import DQNAgent


class Trainer:
    """
    Gestor del entrenamiento del agente.
    
    Args:
        episodes: Número de episodios a entrenar
        train_frequency: Entrenar cada N frames (1)
        save_frequency: Guardar checkpoint cada N episodios (100)
        fast_mode: Entrenar a máxima velocidad sin visualización
    """
    
    def __init__(
        self,
        episodes: int = 10000,
        train_frequency: int = 1,  # Entrenar cada frame
        save_frequency: int = 100,
        fast_mode: bool = True
    ):
        self.episodes = episodes
        self.train_frequency = train_frequency
        self.save_frequency = save_frequency
        self.fast_mode = fast_mode
        
        # Inicializar entorno y agente
        self.env = FlappyBirdEnv()
        self.agent = DQNAgent()
        
        # Métricas
        self.episode_scores: List[int] = []
        self.episode_losses: List[float] = []
        self.episode_epsilons: List[float] = []
        self.best_score = 0
        
        # Estado del entrenamiento
        self.is_training = False
        self.current_episode = 0
        
        # Directorios
        self.models_dir = "models"
        self.logs_dir = "logs"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def train_episode(self) -> Dict:
        """
        Entrena un episodio completo.
        
        Returns:
            Métricas del episodio
        """
        state = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        while not self.env.done:
            # Seleccionar acción
            action = self.agent.select_action(state)
            
            # Ejecutar acción
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            # Almacenar experiencia
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Entrenar
            if self.env.frames % self.train_frequency == 0:
                loss = self.agent.train_step()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
            
            state = next_state
        
        # Calcular métricas
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        score = info['score']
        
        # Actualizar best score
        if score > self.best_score:
            self.best_score = score
            self.save_best_model()
        
        metrics = {
            'episode': self.current_episode,
            'score': score,
            'epsilon': self.agent.epsilon,
            'loss': avg_loss,
            'avg_q_value': self.agent.avg_q_value,
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def train(self) -> None:
        """Ejecuta el loop de entrenamiento completo."""
        self.is_training = True
        print(f"Iniciando entrenamiento por {self.episodes} episodios...")
        
        for episode in range(1, self.episodes + 1):
            self.current_episode = episode
            
            # Entrenar episodio
            metrics = self.train_episode()
            
            # Guardar métricas
            self.episode_scores.append(metrics['score'])
            self.episode_losses.append(metrics['loss'])
            self.episode_epsilons.append(metrics['epsilon'])
            
            # Log
            if episode % 10 == 0:
                avg_score = sum(self.episode_scores[-10:]) / 10
                print(f"Episodio {episode}/{self.episodes} | "
                      f"Score: {metrics['score']} | "
                      f"Avg Score (10): {avg_score:.1f} | "
                      f"Best: {self.best_score} | "
                      f"Epsilon: {metrics['epsilon']:.3f} | "
                      f"Loss: {metrics['loss']:.4f}")
            
            # Guardar checkpoint
            if episode % self.save_frequency == 0:
                self.save_checkpoint(episode)
                self.save_training_log()
        
        self.is_training = False
        print("Entrenamiento completado!")
    
    def save_checkpoint(self, episode: int) -> None:
        """Guarda un checkpoint del modelo."""
        filepath = os.path.join(self.models_dir, f"checkpoint_episode{episode}.pth")
        self.agent.save_model(filepath)
        print(f"Checkpoint guardado: {filepath}")
    
    def save_best_model(self) -> None:
        """Guarda el mejor modelo."""
        filepath = os.path.join(self.models_dir, "best_model.pth")
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Carga un modelo guardado."""
        self.agent.load_model(filepath)
        print(f"Modelo cargado: {filepath}")
    
    def save_training_log(self) -> None:
        """Guarda el log de entrenamiento en JSON."""
        log_data = {
            'episodes': [
                {
                    'episode': i + 1,
                    'score': self.episode_scores[i],
                    'epsilon': self.episode_epsilons[i],
                    'loss': self.episode_losses[i]
                }
                for i in range(len(self.episode_scores))
            ],
            'best_score': self.best_score
        }
        
        filepath = os.path.join(self.logs_dir, "training_log.json")
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)


if __name__ == "__main__":
    # Entrenamiento standalone
    trainer = Trainer(episodes=1000, fast_mode=True)
    trainer.train()
