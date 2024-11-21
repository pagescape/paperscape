import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import spacy
import nltk

from services.paper_processor import PaperProcessor
from services.game_generator import GameGenerator
from services.game_state import game_state_manager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Custom Jinja2 filters
@app.template_filter('datetime')
def format_datetime(value):
    """Format datetime string to readable format"""
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime('%b %d, %Y %I:%M %p')
    except (ValueError, TypeError):
        return value

@app.template_filter('duration')
def format_duration(seconds):
    """Format seconds to readable duration"""
    try:
        seconds = float(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    except (ValueError, TypeError):
        return "N/A"

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_paper():
    """Handle research paper upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        return jsonify({'error': 'Only PDF and TXT files are supported'}), 400
    
    try:
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Process the paper
        processor = PaperProcessor()  # Initialize without filepath
        result = processor.process_paper(filepath)  # Process with filepath
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            'message': 'Paper processed successfully',
            'result': result
        }), 200
    
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error processing paper: {str(e)}")
        return jsonify({'error': 'Failed to process the paper. Please try again.'}), 500

@app.route('/play-game/<game_id>')
def play_game(game_id):
    """Render a specific game"""
    # Retrieve and render game based on game_id
    return render_template('game.html', game_id=game_id)

@app.route('/progress')
def view_progress():
    """View user's learning progress and achievements"""
    # For demo purposes, using a fixed user_id
    user_id = "demo_user"
    
    # Get user's progress data
    progress_data = game_state_manager.get_user_progress(user_id)
    achievements = game_state_manager.achievements.get(user_id, [])
    
    # Calculate statistics
    total_games = len(progress_data)
    total_completed = sum(1 for game in progress_data.values() if game.get('completion_percentage', 0) == 100)
    total_points = sum(game.get('total_points', 0) for game in progress_data.values())
    
    # Prepare chart data
    chart_dates = []
    chart_points = []
    current_date = datetime.now()
    
    for i in range(7):  # Last 7 days
        date = current_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        points = sum(
            game.get('total_points', 0)
            for game in progress_data.values()
            if game.get('started_at', '').startswith(date_str)
        )
        chart_dates.insert(0, date.strftime('%b %d'))
        chart_points.insert(0, points)
    
    # Prepare knowledge map data
    knowledge_nodes = []
    knowledge_links = []
    node_id = 0
    
    for game_id, game_data in progress_data.items():
        # Add game node
        game_node = {
            'id': f'game_{node_id}',
            'label': game_data.get('type', 'Unknown Game'),
            'group': 'game'
        }
        knowledge_nodes.append(game_node)
        
        # Add achievement nodes and links
        for achievement in game_data.get('achievements', []):
            node_id += 1
            achievement_node = {
                'id': f'achievement_{node_id}',
                'label': achievement.get('name', 'Unknown Achievement'),
                'group': 'achievement'
            }
            knowledge_nodes.append(achievement_node)
            knowledge_links.append({
                'source': game_node['id'],
                'target': achievement_node['id']
            })
        
        node_id += 1
    
    return render_template('progress.html',
        total_games=total_games,
        total_completed=total_completed,
        total_points=total_points,
        recent_achievements=achievements[-5:],  # Show last 5 achievements
        game_history=sorted(
            [{'id': k, **v} for k, v in progress_data.items()],
            key=lambda x: x.get('started_at', ''),
            reverse=True
        ),
        chart_dates=chart_dates,
        chart_points=chart_points,
        knowledge_nodes=knowledge_nodes,
        knowledge_links=knowledge_links
    )

if __name__ == '__main__':
    # Development server
    app.run(debug=True)
else:
    # Production server
    app.config['DEBUG'] = False
    app.config['UPLOAD_FOLDER'] = '/home/yourusername/research-game-platform/uploads'  # Update this path on PythonAnywhere
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
