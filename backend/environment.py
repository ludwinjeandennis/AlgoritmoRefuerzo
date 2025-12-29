"""
Entorno de Flappy Bird para entrenamiento de RL
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
import random


class FlappyBirdEnv:
    """
    Simulación del juego Flappy Bird.
    
    Estado: [distancia_horizontal_tubería, distancia_vertical_tubería, velocidad_pajaro]
    Acciones: 0=no hacer nada, 1=saltar
    Recompensas: +1 por frame, +10 por pasar tubería, -1000 por colisión
    """
    
    def __init__(self, width: int = 400, height: int = 600):
        # Dimensiones del canvas
        self.width = width
        self.height = height
        
        # Configuración del pájaro (ajustada para facilitar aprendizaje)
        self.bird_x = 80
        self.bird_y = 0
        self.bird_radius = 12
        self.bird_velocity = 0
        
        # Física ajustada para aprendizaje más eficiente
        self.gravity = 0.5  # Aceleración hacia abajo por frame
        self.jump_strength = -7  # Salto más bajo (antes -9, ahora -7)
        self.max_velocity = 10  # Velocidad máxima de caída
        
        # Configuración de tuberías (más fácil para aprender)
        self.pipe_width = 52
        self.pipe_gap = 150  # Gap más amplio (antes 120, ahora 150)
        self.pipe_velocity = 3  # Velocidad más rápida (antes 2, ahora 3)
        self.pipe_spawn_distance = 250  # Tuberías más separadas (antes 200, ahora 250)
        self.pipes: List[Dict] = []
        
        # Configuración del suelo
        self.ground_height = 100
        self.ground_y = self.height - self.ground_height
        
        # Estado del juego
        self.score = 0
        self.frames = 0
        self.done = False
        
        # Inicializar
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reinicia el juego.
        
        Returns:
            Estado inicial
        """
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.score = 0
        self.frames = 0
        self.done = False
        
        # Crear tuberías iniciales
        self.pipes = []
        for i in range(3):
            self._spawn_pipe(self.width + i * self.pipe_spawn_distance)
        
        return self.get_state()
    
    def _spawn_pipe(self, x: float) -> None:
        """Crea una nueva tubería."""
        # Altura aleatoria del gap
        min_gap_y = 100
        max_gap_y = self.ground_y - self.pipe_gap - 100
        gap_y = random.randint(min_gap_y, max_gap_y)
        
        pipe = {
            'x': x,
            'gap_y': gap_y,
            'passed': False
        }
        self.pipes.append(pipe)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: 0=no hacer nada, 1=saltar
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.done:
            return self.get_state(), 0, True, {}
        
        # Aplicar acción
        if action == 1:
            self.bird_velocity = self.jump_strength
        
        # Física del pájaro
        self.bird_velocity += self.gravity
        # Limitar velocidad máxima de caída
        self.bird_velocity = min(self.bird_velocity, self.max_velocity)
        self.bird_y += self.bird_velocity
        
        # Mover tuberías
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_velocity
        
        # Eliminar tuberías fuera de pantalla y crear nuevas
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
            self._spawn_pipe(self.pipes[-1]['x'] + self.pipe_spawn_distance)
        
        # Calcular recompensa según especificaciones
        # +1 por cada frame que sobrevive
        reward = 1.0
        
        # Shaped reward sutil: pequeño bonus por estar cerca del centro del gap
        # Esto ayuda al aprendizaje sin dominar las recompensas principales
        next_pipe = self._get_next_pipe()
        if next_pipe:
            gap_center = next_pipe['gap_y'] + self.pipe_gap / 2
            vertical_dist = abs(self.bird_y - gap_center)
            # Bonus muy pequeño (máximo +0.1) para no interferir con recompensas principales
            reward += max(0, (1 - vertical_dist / 150)) * 0.1
        
        # +10 por pasar exitosamente entre tuberías
        for pipe in self.pipes:
            if not pipe['passed'] and pipe['x'] + self.pipe_width < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                reward += 10
        
        # -1000 por colisionar con tuberías, suelo o techo
        collision = self._check_collision()
        if collision:
            reward = -1000
            self.done = True
        
        self.frames += 1
        
        info = {
            'score': self.score,
            'frames': self.frames
        }
        
        return self.get_state(), reward, self.done, info
    
    def _get_next_pipe(self) -> Optional[Dict]:
        """Obtiene la tubería más cercana por delante."""
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                return pipe
        return None
    
    def _check_collision(self) -> bool:
        """Verifica colisiones con tuberías, techo y suelo."""
        # Colisión con techo o suelo
        if self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= self.ground_y:
            return True
        
        # Colisión con tuberías
        for pipe in self.pipes:
            # Verificar si el pájaro está en el rango horizontal de la tubería
            if (self.bird_x + self.bird_radius > pipe['x'] and 
                self.bird_x - self.bird_radius < pipe['x'] + self.pipe_width):
                
                # Verificar si está fuera del gap
                if (self.bird_y - self.bird_radius < pipe['gap_y'] or 
                    self.bird_y + self.bird_radius > pipe['gap_y'] + self.pipe_gap):
                    return True
        
        return False
    
    def get_state(self) -> np.ndarray:
        """
        Obtiene el estado actual del juego.
        
        Returns:
            [distancia_horizontal, distancia_vertical, velocidad, altura_normalizada, distancia_suelo]
        """
        # Encontrar la tubería más cercana
        next_pipe = self._get_next_pipe()
        
        if next_pipe is None:
            # No hay tuberías (no debería pasar)
            horizontal_dist = 0
            vertical_dist = 0
        else:
            # Distancia horizontal al borde de la tubería
            horizontal_dist = next_pipe['x'] - self.bird_x
            
            # Distancia vertical al centro del gap
            gap_center = next_pipe['gap_y'] + self.pipe_gap / 2
            vertical_dist = self.bird_y - gap_center
        
        # Normalizar valores para mejor aprendizaje
        state = np.array([
            horizontal_dist / self.width,  # Distancia horizontal normalizada
            vertical_dist / self.height,   # Distancia vertical normalizada
            self.bird_velocity / 10,       # Velocidad normalizada
            self.bird_y / self.height,     # Altura del pájaro normalizada
            (self.ground_y - self.bird_y) / self.height  # Distancia al suelo normalizada
        ], dtype=np.float32)
        
        return state
    
    def render(self) -> Dict:
        """
        Genera datos para visualización.
        
        Returns:
            Diccionario con datos del juego para el frontend
        """
        return {
            'bird': {
                'x': self.bird_x,
                'y': self.bird_y,
                'radius': self.bird_radius,
                'velocity': self.bird_velocity
            },
            'pipes': [
                {
                    'x': pipe['x'],
                    'gap_y': pipe['gap_y'],
                    'gap_height': self.pipe_gap,
                    'width': self.pipe_width
                }
                for pipe in self.pipes
            ],
            'score': self.score,
            'done': self.done,
            'ground_y': self.ground_y
        }
