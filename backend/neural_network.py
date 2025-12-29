"""
Red Neuronal Deep Q-Network (DQN) para Flappy Bird
Arquitectura: 5 -> 128 -> 128 -> 64 -> 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQN(nn.Module):
    """
    Red neuronal profunda para aproximar la función Q.
    
    Args:
        input_size: Tamaño del estado de entrada (5: distancia_h, distancia_v, velocidad, altura, dist_suelo)
        hidden_size: Número de neuronas en capas ocultas
        output_size: Número de acciones posibles (2: no hacer nada, saltar)
    """
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128, output_size: int = 2):
        super(DQN, self).__init__()
        
        # Capas de la red (arquitectura mejorada)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, output_size)
        
        # Inicialización Xavier para mejor convergencia
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Inicializa los pesos usando Xavier uniform."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante.
        
        Args:
            x: Tensor de estado [batch_size, 5]
            
        Returns:
            Q-values para cada acción [batch_size, 2]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Sin activación en la salida
        return x
