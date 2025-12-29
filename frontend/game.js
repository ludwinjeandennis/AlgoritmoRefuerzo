/**
 * Flappy Bird RL - Frontend JavaScript
 * Renderizado Canvas y comunicación WebSocket
 */

// Configuración
const CONFIG = {
    WS_URL: 'ws://localhost:8000/ws',
    CANVAS_WIDTH: 400,
    CANVAS_HEIGHT: 600,
    FPS: 60
};

// Estado global
let ws = null;
let gameState = null;
let metrics = null;
let isConnected = false;

// Canvas
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Elementos UI
const elements = {
    startBtn: document.getElementById('startBtn'),
    pauseBtn: document.getElementById('pauseBtn'),
    saveBtn: document.getElementById('saveBtn'),
    loadBtn: document.getElementById('loadBtn'),
    episodeCount: document.getElementById('episodeCount'),
    currentScore: document.getElementById('currentScore'),
    bestScore: document.getElementById('bestScore'),
    epsilon: document.getElementById('epsilon'),
    loss: document.getElementById('loss'),
    qValue: document.getElementById('qValue'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    gameOverlay: document.getElementById('gameOverlay'),
    modelName: document.getElementById('modelName')  // Nombre del modelo
};

// Gráficas Chart.js
let scoreChart, lossChart, epsilonChart;

// Sistema de partículas
let particles = [];

class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 4;
        this.vy = (Math.random() - 0.5) * 4;
        this.life = 1.0;
        this.decay = 0.02;
        this.size = Math.random() * 3 + 2;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= this.decay;
    }

    draw(ctx) {
        ctx.fillStyle = `rgba(255, 204, 0, ${this.life})`;
        ctx.fillRect(this.x, this.y, this.size, this.size);
    }

    isDead() {
        return this.life <= 0;
    }
}

// Inicialización
function init() {
    initCharts();
    connectWebSocket();
    setupEventListeners();
    startRenderLoop();
}

// WebSocket
function connectWebSocket() {
    console.log('Conectando a WebSocket...');

    ws = new WebSocket(CONFIG.WS_URL);

    ws.onopen = () => {
        console.log('WebSocket conectado');
        isConnected = true;
        updateConnectionStatus(true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        gameState = data.game;
        metrics = data.metrics;

        updateUI();
        updateCharts(data.history);
    };

    ws.onerror = (error) => {
        console.error('Error WebSocket:', error);
        updateConnectionStatus(false);
    };

    ws.onclose = () => {
        console.log('WebSocket desconectado');
        isConnected = false;
        updateConnectionStatus(false);

        // Reconectar después de 3 segundos
        setTimeout(connectWebSocket, 3000);
    };
}

function sendCommand(action, data = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action, ...data }));
    }
}

// Event Listeners
function setupEventListeners() {
    elements.startBtn.addEventListener('click', () => sendCommand('start'));
    elements.pauseBtn.addEventListener('click', () => sendCommand('pause'));
    elements.saveBtn.addEventListener('click', () => sendCommand('save'));
    elements.loadBtn.addEventListener('click', () => openModelSelector());
}

// Modal de selección de modelos
async function openModelSelector() {
    const modal = document.getElementById('modelModal');
    const modalClose = document.getElementById('modalClose');
    const modelList = document.getElementById('modelList');

    // Mostrar modal
    modal.classList.add('show');

    // Cerrar modal al hacer clic en X
    modalClose.onclick = () => modal.classList.remove('show');

    // Cerrar modal al hacer clic fuera
    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    };

    // Cargar lista de modelos
    try {
        const response = await fetch('http://localhost:8000/models');
        const data = await response.json();

        if (data.models && data.models.length > 0) {
            modelList.innerHTML = data.models.map(model => `
                <div class="model-item" onclick="loadModel('${model.filepath}')">
                    <div class="model-item-name">📦 ${model.filename}</div>
                    <div class="model-item-info">
                        Tamaño: ${model.size_kb} KB | 
                        Modificado: ${new Date(model.modified * 1000).toLocaleString()}
                    </div>
                </div>
            `).join('');
        } else {
            modelList.innerHTML = '<p class="no-models">No hay modelos guardados</p>';
        }
    } catch (error) {
        console.error('Error al cargar modelos:', error);
        modelList.innerHTML = '<p class="no-models">Error al cargar modelos</p>';
    }
}

function loadModel(filepath) {
    // Cerrar modal
    document.getElementById('modelModal').classList.remove('show');

    // Enviar comando de carga con la ruta del archivo
    sendCommand('load', { filepath: filepath });

    console.log('Cargando modelo:', filepath);
}

// Actualizar UI
function updateUI() {
    if (!metrics) return;

    elements.episodeCount.textContent = metrics.episode;
    elements.currentScore.textContent = metrics.score;
    elements.bestScore.textContent = metrics.best_score;
    elements.epsilon.textContent = metrics.epsilon.toFixed(3);
    elements.loss.textContent = metrics.loss.toFixed(4);
    elements.qValue.textContent = metrics.avg_q_value.toFixed(2);

    // Actualizar nombre del modelo
    if (metrics.model_name) {
        elements.modelName.textContent = metrics.model_name;
    }

    // NO mostrar overlay de Game Over para mejor visualización del aprendizaje continuo
    // if (gameState && gameState.done) {
    //     elements.gameOverlay.classList.add('show');
    // } else {
    //     elements.gameOverlay.classList.remove('show');
    // }
}

function updateConnectionStatus(connected) {
    if (connected) {
        elements.statusDot.classList.add('connected');
        elements.statusText.textContent = 'Conectado';
    } else {
        elements.statusDot.classList.remove('connected');
        elements.statusText.textContent = 'Desconectado';
    }
}

// Renderizado Canvas
function render() {
    if (!gameState) {
        drawWaitingScreen();
        return;
    }

    // Limpiar canvas
    ctx.clearRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

    // Dibujar elementos
    drawBackground();
    drawPipes();
    drawBird();
    drawGround();
    drawScore();
    drawParticles();
}

function drawWaitingScreen() {
    ctx.fillStyle = '#70c5ce';
    ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

    ctx.fillStyle = '#fff';
    ctx.font = '16px "Press Start 2P"';
    ctx.textAlign = 'center';
    ctx.fillText('Esperando...', CONFIG.CANVAS_WIDTH / 2, CONFIG.CANVAS_HEIGHT / 2);
}

function drawBackground() {
    // Cielo degradado
    const gradient = ctx.createLinearGradient(0, 0, 0, CONFIG.CANVAS_HEIGHT);
    gradient.addColorStop(0, '#70c5ce');
    gradient.addColorStop(1, '#a8e6f0');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

    // Nubes simples
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    const cloudOffset = (Date.now() / 50) % CONFIG.CANVAS_WIDTH;
    for (let i = 0; i < 3; i++) {
        const x = (i * 150 - cloudOffset) % CONFIG.CANVAS_WIDTH;
        drawCloud(x, 50 + i * 80);
    }
}

function drawCloud(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.arc(x + 15, y, 25, 0, Math.PI * 2);
    ctx.arc(x + 30, y, 20, 0, Math.PI * 2);
    ctx.fill();
}

function drawBird() {
    const bird = gameState.bird;

    // Crear partículas al saltar
    if (bird.velocity < -5) {
        particles.push(new Particle(bird.x, bird.y));
    }

    // Cuerpo del pájaro
    ctx.fillStyle = '#ffcc00';
    ctx.beginPath();
    ctx.arc(bird.x, bird.y, bird.radius, 0, Math.PI * 2);
    ctx.fill();

    // Borde
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Ojo
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(bird.x + 5, bird.y - 3, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(bird.x + 6, bird.y - 3, 2, 0, Math.PI * 2);
    ctx.fill();

    // Pico
    ctx.fillStyle = '#ff6600';
    ctx.beginPath();
    ctx.moveTo(bird.x + bird.radius, bird.y);
    ctx.lineTo(bird.x + bird.radius + 8, bird.y - 2);
    ctx.lineTo(bird.x + bird.radius, bird.y + 2);
    ctx.fill();
}

function drawPipes() {
    gameState.pipes.forEach(pipe => {
        // Tubería superior
        ctx.fillStyle = '#5ac54f';
        ctx.fillRect(pipe.x, 0, pipe.width, pipe.gap_y);

        // Borde oscuro
        ctx.strokeStyle = '#2d6f2d';
        ctx.lineWidth = 3;
        ctx.strokeRect(pipe.x, 0, pipe.width, pipe.gap_y);

        // Highlight
        ctx.fillStyle = '#7ed957';
        ctx.fillRect(pipe.x + 5, 0, 10, pipe.gap_y);

        // Tubería inferior
        const bottomY = pipe.gap_y + pipe.gap_height;
        const bottomHeight = CONFIG.CANVAS_HEIGHT - bottomY - gameState.ground_y + CONFIG.CANVAS_HEIGHT;

        ctx.fillStyle = '#5ac54f';
        ctx.fillRect(pipe.x, bottomY, pipe.width, bottomHeight);

        ctx.strokeStyle = '#2d6f2d';
        ctx.strokeRect(pipe.x, bottomY, pipe.width, bottomHeight);

        ctx.fillStyle = '#7ed957';
        ctx.fillRect(pipe.x + 5, bottomY, 10, bottomHeight);
    });
}

function drawGround() {
    const groundY = gameState.ground_y;

    // Suelo
    ctx.fillStyle = '#ded895';
    ctx.fillRect(0, groundY, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT - groundY);

    // Textura del suelo
    ctx.strokeStyle = '#c4b86a';
    ctx.lineWidth = 2;
    const offset = (Date.now() / 20) % 20;
    for (let x = -offset; x < CONFIG.CANVAS_WIDTH; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, groundY);
        ctx.lineTo(x + 10, groundY + 10);
        ctx.stroke();
    }
}

function drawScore() {
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 3;
    ctx.font = '32px "Press Start 2P"';
    ctx.textAlign = 'center';

    const scoreText = gameState.score.toString();
    ctx.strokeText(scoreText, CONFIG.CANVAS_WIDTH / 2, 60);
    ctx.fillText(scoreText, CONFIG.CANVAS_WIDTH / 2, 60);
}

function drawParticles() {
    particles = particles.filter(p => !p.isDead());
    particles.forEach(p => {
        p.update();
        p.draw(ctx);
    });
}

// Loop de renderizado
function startRenderLoop() {
    setInterval(() => {
        render();
    }, 1000 / CONFIG.FPS);
}

// Gráficas
function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                display: false,
                grid: { display: false }
            },
            y: {
                ticks: { color: '#a0a0a0', font: { size: 8 } },
                grid: { color: 'rgba(255, 255, 255, 0.1)' }
            }
        }
    };

    // Score Chart
    scoreChart = new Chart(document.getElementById('scoreChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Score',
                data: [],
                borderColor: '#ffcc00',
                backgroundColor: 'rgba(255, 204, 0, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // Loss Chart
    lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#e94560',
                backgroundColor: 'rgba(233, 69, 96, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // Epsilon Chart
    epsilonChart = new Chart(document.getElementById('epsilonChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Epsilon',
                data: [],
                borderColor: '#4ecdc4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });
}

function updateCharts(history) {
    if (!history) return;

    const maxPoints = 100;

    // Score Chart
    if (history.scores && history.scores.length > 0) {
        scoreChart.data.labels = history.scores.map((_, i) => i);
        scoreChart.data.datasets[0].data = history.scores.slice(-maxPoints);
        scoreChart.update('none');
    }

    // Loss Chart
    if (history.losses && history.losses.length > 0) {
        lossChart.data.labels = history.losses.map((_, i) => i);
        lossChart.data.datasets[0].data = history.losses.slice(-maxPoints);
        lossChart.update('none');
    }

    // Epsilon Chart
    if (history.epsilons && history.epsilons.length > 0) {
        epsilonChart.data.labels = history.epsilons.map((_, i) => i);
        epsilonChart.data.datasets[0].data = history.epsilons.slice(-maxPoints);
        epsilonChart.update('none');
    }
}

// Iniciar aplicación
window.addEventListener('DOMContentLoaded', init);
