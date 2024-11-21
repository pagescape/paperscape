// Game State Management
let gameState = {
    points: 0,
    discoveries: [],
    currentStage: 0,
    inventory: [],
    achievements: []
};

// Simulation Game Functions
class SimulationManager {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.config = config;
        this.engine = Matter.Engine.create();
        this.setupSimulation();
    }

    setupSimulation() {
        this.render = Matter.Render.create({
            canvas: this.canvas,
            engine: this.engine,
            options: {
                width: this.canvas.width,
                height: this.canvas.height,
                wireframes: false,
                background: '#1a1a2e'
            }
        });

        // Add particles
        this.particles = this.config.elements.map((element, index) => {
            return Matter.Bodies.circle(
                Math.random() * this.render.options.width,
                Math.random() * this.render.options.height,
                20,
                {
                    render: {
                        fillStyle: `hsl(${index * 60}, 80%, 60%)`
                    },
                    label: element
                }
            );
        });

        Matter.World.add(this.engine.world, this.particles);
        Matter.Engine.run(this.engine);
        Matter.Render.run(this.render);
    }

    updateParameters(params) {
        this.particles.forEach(particle => {
            Matter.Body.setVelocity(particle, {
                x: (Math.random() - 0.5) * params.speed,
                y: (Math.random() - 0.5) * params.speed
            });
        });
    }
}

// Story Adventure Functions
class StoryManager {
    constructor(container, story) {
        this.container = container;
        this.story = story;
        this.currentScene = 0;
    }

    renderScene() {
        const scene = this.story.chapters[this.currentScene];
        this.container.innerHTML = `
            <div class="story-scene fade-in" 
                style="background-image: url('/static/backgrounds/${scene.scene.background}.jpg')">
                <h2 class="slide-up">${scene.title}</h2>
                <p class="mt-4 slide-up">${scene.narrative}</p>
                ${this.renderChoices(scene.choices)}
            </div>
        `;
        this.animateCharacters(scene.scene.characters);
    }

    renderChoices(choices) {
        if (!choices) return '';
        return `
            <div class="choices-container mt-5">
                ${choices.map((choice, index) => `
                    <button class="btn btn-light mb-2 w-100 slide-up" 
                        onclick="storyManager.makeChoice(${choice.leads_to})"
                        style="animation-delay: ${index * 0.2}s">
                        ${choice.text}
                    </button>
                `).join('')}
            </div>
        `;
    }

    animateCharacters(characters) {
        characters.forEach((char, index) => {
            const charElement = document.createElement('div');
            charElement.className = `character fade-in`;
            charElement.style.backgroundImage = `url('/static/sprites/${char.sprite}.png')`;
            charElement.style[char.position] = '20px';
            charElement.style.animationDelay = `${index * 0.3}s`;
            this.container.querySelector('.story-scene').appendChild(charElement);
        });
    }

    makeChoice(nextScene) {
        gameState.currentStage = nextScene - 1;
        this.currentScene = nextScene - 1;
        this.renderScene();
    }
}

// Concept Map Functions
class ConceptMapManager {
    constructor(container, conceptMap) {
        this.container = container;
        this.conceptMap = conceptMap;
        this.setupForceGraph();
    }

    setupForceGraph() {
        const width = this.container.offsetWidth;
        const height = 600;

        const svg = d3.select(this.container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const simulation = d3.forceSimulation(this.conceptMap.nodes)
            .force('link', d3.forceLink(this.conceptMap.relationships).id(d => d.id))
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(width / 2, height / 2));

        this.drawLinks(svg, simulation);
        this.drawNodes(svg, simulation);
    }

    drawLinks(svg, simulation) {
        const links = svg.append('g')
            .selectAll('line')
            .data(this.conceptMap.relationships)
            .enter().append('line')
            .style('stroke', '#999')
            .style('stroke-opacity', 0.6);

        simulation.on('tick', () => {
            links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        });
    }

    drawNodes(svg, simulation) {
        const nodes = svg.append('g')
            .selectAll('circle')
            .data(this.conceptMap.nodes)
            .enter().append('circle')
            .attr('r', 20)
            .style('fill', d => this.conceptMap.visualization.node_colors[d.type])
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }));
    }
}

// Research Journey Functions
class ResearchJourneyManager {
    constructor(container, journey) {
        this.container = container;
        this.journey = journey;
        this.currentStage = 0;
    }

    renderJourney() {
        this.container.innerHTML = `
            <div class="research-journey fade-in">
                ${this.renderProgressBar()}
                ${this.renderStages()}
                ${this.renderAchievements()}
            </div>
        `;
    }

    renderProgressBar() {
        const progress = (this.currentStage / this.journey.stages.length) * 100;
        return `
            <div class="progress mb-4">
                <div class="progress-bar bg-success" role="progressbar" 
                    style="width: ${progress}%">
                </div>
            </div>
        `;
    }

    renderStages() {
        return this.journey.stages.map((stage, index) => `
            <div class="research-journey-stage ${index === this.currentStage ? 'active' : ''}"
                style="opacity: ${index <= this.currentStage ? 1 : 0.5}">
                <h3>${stage.title}</h3>
                <p>${stage.description}</p>
                ${this.renderStageInteraction(stage.interaction)}
            </div>
        `).join('');
    }

    renderStageInteraction(interaction) {
        switch(interaction.type) {
            case 'hypothesis_builder':
                return this.renderHypothesisBuilder(interaction);
            case 'lab_simulation':
                return this.renderLabSimulation(interaction);
            default:
                return '';
        }
    }

    renderHypothesisBuilder(interaction) {
        return `
            <div class="hypothesis-builder">
                <select class="form-select mb-2">
                    ${interaction.components.map(comp => `
                        <option value="${comp}">${comp}</option>
                    `).join('')}
                </select>
                <button class="btn btn-primary" onclick="journeyManager.submitHypothesis()">
                    Form Hypothesis
                </button>
            </div>
        `;
    }

    renderLabSimulation(interaction) {
        return `
            <div class="lab-simulation">
                <div class="equipment-selection mb-3">
                    ${interaction.equipment.map(equip => `
                        <button class="btn btn-outline-primary m-1">
                            <i class="fas fa-${equip}"></i> ${equip}
                        </button>
                    `).join('')}
                </div>
                <div class="variables-control">
                    ${interaction.variables.map(variable => `
                        <div class="form-group">
                            <label>${variable.name}</label>
                            <input type="range" class="form-range" 
                                min="${variable.range[0]}" 
                                max="${variable.range[1]}"
                                value="${(variable.range[0] + variable.range[1]) / 2}">
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    renderAchievements() {
        return `
            <div class="achievements-section mt-4">
                <h4>Achievements</h4>
                <div class="d-flex flex-wrap">
                    ${this.journey.achievements.map(achievement => `
                        <div class="achievement-badge">
                            <i class="fas fa-trophy"></i>
                            ${achievement.name}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    submitHypothesis() {
        // Handle hypothesis submission
        this.currentStage++;
        this.renderJourney();
    }
}

// Initialize game based on type
function initializeGame(gameData) {
    const container = document.getElementById('game-interface');
    
    switch(gameData.type) {
        case 'simulation':
            return new SimulationManager(
                document.getElementById('simulation-canvas'),
                gameData
            );
        case 'story_adventure':
            return new StoryManager(container, gameData);
        case 'concept_map':
            return new ConceptMapManager(container, gameData);
        case 'research_journey':
            return new ResearchJourneyManager(container, gameData);
        default:
            console.error('Unknown game type:', gameData.type);
            return null;
    }
}

class GameManager {
    constructor(config) {
        this.paperId = config.paperId;
        this.gameType = config.gameType;
        this.difficulty = config.difficulty;
        this.score = 0;
        this.progress = 0;
        this.gameState = 'ready';
        this.socket = io();
        this.initializePhaser();
    }

    initializePhaser() {
        const config = {
            type: Phaser.AUTO,
            parent: 'game-canvas',
            width: 800,
            height: 600,
            physics: {
                default: 'arcade',
                arcade: {
                    gravity: { y: 300 },
                    debug: false
                }
            },
            scene: {
                preload: this.preload.bind(this),
                create: this.create.bind(this),
                update: this.update.bind(this)
            }
        };
        this.game = new Phaser.Game(config);
    }

    preload() {
        // Load game assets
        this.game.load.image('researcher', '/static/sprites/researcher.png');
        this.game.load.image('platform', '/static/sprites/platform.png');
        this.game.load.image('concept', '/static/sprites/concept.png');
    }

    create() {
        // Set up game world
        this.platforms = this.physics.add.staticGroup();
        this.concepts = this.physics.add.group();
        this.player = this.physics.add.sprite(100, 450, 'researcher');
        
        // Add colliders
        this.physics.add.collider(this.player, this.platforms);
        this.physics.add.overlap(this.player, this.concepts, this.collectConcept, null, this);
        
        // Set up controls
        this.cursors = this.input.keyboard.createCursorKeys();
        
        // Initialize game elements based on paper content
        this.initializeGameElements();
    }

    update() {
        if (this.gameState === 'playing') {
            this.handlePlayerMovement();
            this.updateProgress();
        }
    }

    handlePlayerMovement() {
        if (this.cursors.left.isDown) {
            this.player.setVelocityX(-160);
        } else if (this.cursors.right.isDown) {
            this.player.setVelocityX(160);
        } else {
            this.player.setVelocityX(0);
        }

        if (this.cursors.up.isDown && this.player.body.touching.down) {
            this.player.setVelocityY(-330);
        }
    }

    initializeGameElements() {
        // Load paper concepts and create game elements
        fetch(`/api/paper/${this.paperId}/concepts`)
            .then(response => response.json())
            .then(data => {
                this.concepts = data;
                this.createGameElements();
            });
    }

    createGameElements() {
        switch(this.gameType) {
            case 'quiz':
                this.initializeQuiz();
                break;
            case 'simulation':
                this.initializeSimulation();
                break;
            case 'puzzle':
                this.initializePuzzle();
                break;
        }
    }

    initializeQuiz() {
        const quizContainer = document.getElementById('quiz-container');
        this.concepts.forEach(concept => {
            const question = this.createQuizQuestion(concept);
            quizContainer.appendChild(question);
        });
    }

    initializeSimulation() {
        const simContainer = document.getElementById('simulation-container');
        // Create interactive simulation based on paper concepts
        this.simulation = new SimulationManager(this.concepts, simContainer);
    }

    initializePuzzle() {
        const puzzleContainer = document.getElementById('puzzle-container');
        // Create drag-and-drop puzzle elements
        this.puzzle = new PuzzleManager(this.concepts, puzzleContainer);
    }

    updateProgress() {
        const progressElement = document.getElementById('paper-progress');
        progressElement.style.width = `${this.progress}%`;
        
        if (this.progress >= 100) {
            this.completeGame();
        }
    }

    completeGame() {
        this.gameState = 'completed';
        this.saveProgress();
        this.showCompletionDialog();
    }

    saveProgress() {
        const progressData = {
            paperId: this.paperId,
            gameType: this.gameType,
            score: this.score,
            completedAt: new Date().toISOString()
        };

        fetch('/api/progress/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(progressData)
        });
    }

    showCompletionDialog() {
        // Show completion message and achievements
        const achievements = this.calculateAchievements();
        this.updateAchievementDisplay(achievements);
    }
}

class SimulationManager {
    constructor(concepts, container) {
        this.concepts = concepts;
        this.container = container;
        this.particles = [];
        this.initialize();
    }

    initialize() {
        this.setupCanvas();
        this.createParticles();
        this.startAnimation();
    }

    setupCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
    }

    createParticles() {
        this.concepts.forEach(concept => {
            const particle = {
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: 5,
                speed: 2,
                direction: Math.random() * Math.PI * 2,
                concept: concept
            };
            this.particles.push(particle);
        });
    }

    startAnimation() {
        const animate = () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.updateParticles();
            this.drawParticles();
            requestAnimationFrame(animate);
        };
        animate();
    }

    updateParticles() {
        this.particles.forEach(particle => {
            particle.x += Math.cos(particle.direction) * particle.speed;
            particle.y += Math.sin(particle.direction) * particle.speed;

            if (particle.x < 0 || particle.x > this.canvas.width) {
                particle.direction = Math.PI - particle.direction;
            }
            if (particle.y < 0 || particle.y > this.canvas.height) {
                particle.direction = -particle.direction;
            }
        });
    }

    drawParticles() {
        this.particles.forEach(particle => {
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = '#4CAF50';
            this.ctx.fill();
            this.ctx.closePath();

            // Draw concept text
            this.ctx.font = '12px Arial';
            this.ctx.fillStyle = '#333';
            this.ctx.fillText(particle.concept.title, particle.x + 10, particle.y);
        });
    }
}

class PuzzleManager {
    constructor(concepts, container) {
        this.concepts = concepts;
        this.container = container;
        this.initialize();
    }

    initialize() {
        this.createPuzzlePieces();
        this.setupDropZones();
    }

    createPuzzlePieces() {
        this.concepts.forEach(concept => {
            const piece = document.createElement('div');
            piece.className = 'puzzle-piece';
            piece.draggable = true;
            piece.textContent = concept.title;
            piece.addEventListener('dragstart', this.handleDragStart.bind(this));
            this.container.appendChild(piece);
        });
    }

    setupDropZones() {
        const dropZone = document.createElement('div');
        dropZone.className = 'drop-zone';
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));
        this.container.appendChild(dropZone);
    }

    handleDragStart(e) {
        e.dataTransfer.setData('text/plain', e.target.textContent);
    }

    handleDragOver(e) {
        e.preventDefault();
    }

    handleDrop(e) {
        e.preventDefault();
        const data = e.dataTransfer.getData('text/plain');
        this.checkPuzzleSolution(data);
    }

    checkPuzzleSolution(pieceData) {
        // Implement puzzle solution checking logic
        console.log('Checking puzzle piece:', pieceData);
    }
}

// Initialize socket connection for real-time updates
const socket = io();

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('game_update', (data) => {
    console.log('Received game update:', data);
});
