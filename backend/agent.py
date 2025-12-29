"""
Agente Q-Learning con Replay Memory y Target Network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional
from neural_network import DQN


class ReplayMemory:
    """Buffer circular para almacenar experiencias."""
    
    def __init__(self, capacity: int = 50000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Almacena una transición."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Muestrea un batch aleatorio de experiencias."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """
    Agente Deep Q-Learning con Double DQN, epsilon-greedy y target network.
    
    Args:
        state_size: Dimensión del estado (5)
        action_size: Número de acciones (2)
        learning_rate: Tasa de aprendizaje (0.0005)
        gamma: Factor de descuento (0.99)
        epsilon_start: Epsilon inicial (1.0)
        epsilon_end: Epsilon final (0.01)
        epsilon_decay: Tasa de decay (0.995)
        memory_size: Tamaño del replay buffer (50000)
        batch_size: Tamaño del batch (64)
        target_update: Frecuencia de actualización de target network (500)
        warmup_steps: Pasos antes de empezar a entrenar (1000)
    """
    
    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 2,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 50000,
        batch_size: int = 64,
        target_update: int = 500,
        warmup_steps: int = 1000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.warmup_steps = warmup_steps
        self.steps = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Redes: policy y target (arquitectura mejorada)
        self.policy_net = DQN(state_size, 128, action_size).to(self.device)
        self.target_net = DQN(state_size, 128, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador y función de pérdida
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Métricas
        self.last_loss = 0.0
        self.avg_q_value = 0.0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Selecciona una acción usando epsilon-greedy.
        
        Args:
            state: Estado actual [5]
            
        Returns:
            Acción seleccionada (0 o 1)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            self.avg_q_value = q_values.mean().item()
            return q_values.argmax(1).item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Almacena una experiencia en la memoria."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Realiza un paso de entrenamiento con Double DQN.
        
        Returns:
            Pérdida del batch
        """
        # Warm-up: no entrenar hasta tener suficientes experiencias
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return 0.0
        
        # Muestrear batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir a tensores
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q-values actuales
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN: usar policy net para seleccionar acción, target net para evaluar
        with torch.no_grad():
            # Seleccionar mejores acciones con policy network
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # Evaluar Q-values con target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular pérdida
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Actualizar target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target_network()
        
        self.last_loss = loss.item()
        return self.last_loss
    
    def update_target_network(self) -> None:
        """Sincroniza la target network con la policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo.
        
        Args:
            filepath: Ruta del archivo .pth
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo.
        
        Args:
            filepath: Ruta del archivo .pth
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
