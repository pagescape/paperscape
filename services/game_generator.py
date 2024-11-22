import random
import json
from typing import Dict, List, Any

class GameGenerator:
    def __init__(self, paper_data: Dict[str, Any]):
        self.paper_data = paper_data
        self.games = {
            'quiz': self._create_quiz_game,
            'simulation': self._create_simulation_game,
            'puzzle': self._create_puzzle_game
        }

    def generate_game(self, game_type: str) -> Dict[str, Any]:
        """Generate a specific type of game from paper data"""
        if game_type not in self.games:
            raise ValueError(f"Unknown game type: {game_type}")
        
        return self.games[game_type]()

    def _create_quiz_game(self) -> Dict[str, Any]:
        """Create an interactive quiz game"""
        questions = self.paper_data['game_elements']['quiz']
        
        game_data = {
            'type': 'quiz',
            'title': 'Research Paper Quiz Challenge',
            'description': 'Test your understanding of the research paper concepts',
            'questions': self._prepare_quiz_questions(questions),
            'settings': {
                'time_limit': 300,  # 5 minutes
                'points_per_question': 10,
                'passing_score': 70
            }
        }
        
        return game_data

    def _prepare_quiz_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare and randomize quiz questions"""
        prepared_questions = []
        
        for q in questions:
            if q['type'] == 'multiple_choice':
                # Shuffle options
                options = q['options'].copy()
                random.shuffle(options)
                
                prepared_questions.append({
                    'type': 'multiple_choice',
                    'question': q['question'],
                    'options': options,
                    'correct_answer': q['correct_answer'],
                    'points': 10
                })
            elif q['type'] == 'true_false':
                prepared_questions.append({
                    'type': 'true_false',
                    'question': q['question'],
                    'correct_answer': q['correct_answer'],
                    'explanation': q['explanation'],
                    'points': 5
                })
        
        random.shuffle(prepared_questions)
        return prepared_questions

    def _create_simulation_game(self) -> Dict[str, Any]:
        """Create an interactive simulation game"""
        simulation_data = self.paper_data['game_elements']['simulation']
        
        game_data = {
            'type': 'simulation',
            'title': 'Concept Interaction Simulator',
            'description': 'Explore how different concepts interact with each other',
            'simulation_type': simulation_data['type'],
            'elements': self._prepare_simulation_elements(simulation_data['elements']),
            'interactions': simulation_data['interactions'],
            'settings': {
                'gravity': 0.1,
                'friction': 0.02,
                'attraction_strength': 0.5,
                'repulsion_strength': 0.3,
                'particle_speed': 2
            }
        }
        
        return game_data

    def _prepare_simulation_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare simulation elements with enhanced properties"""
        prepared_elements = []
        
        for element in elements:
            prepared_element = element.copy()
            # Add visual properties
            prepared_element['visual'] = {
                'color': self._get_element_color(element['type']),
                'size': element['properties']['size'],
                'shape': self._get_element_shape(element['type'])
            }
            # Add physics properties
            prepared_element['physics'] = {
                'mass': element['properties']['mass'],
                'charge': element['properties']['charge'],
                'initial_velocity': {
                    'x': random.uniform(-1, 1),
                    'y': random.uniform(-1, 1)
                }
            }
            prepared_elements.append(prepared_element)
        
        return prepared_elements

    def _get_element_color(self, element_type: str) -> str:
        """Get color based on element type"""
        colors = {
            'definition': '#4CAF50',
            'methodology': '#2196F3',
            'result': '#FF9800'
        }
        return colors.get(element_type, '#9C27B0')

    def _get_element_shape(self, element_type: str) -> str:
        """Get shape based on element type"""
        shapes = {
            'definition': 'circle',
            'methodology': 'hexagon',
            'result': 'square'
        }
        return shapes.get(element_type, 'circle')

    def _create_puzzle_game(self) -> Dict[str, Any]:
        """Create an interactive puzzle game"""
        puzzle_data = self.paper_data['game_elements']['puzzle']
        
        game_data = {
            'type': 'puzzle',
            'title': 'Concept Connection Challenge',
            'description': 'Connect related concepts to build a knowledge map',
            'puzzle_type': puzzle_data['type'],
            'nodes': self._prepare_puzzle_nodes(puzzle_data['nodes']),
            'connections': puzzle_data['connections'],
            'settings': {
                'grid_size': {'width': 800, 'height': 600},
                'snap_to_grid': True,
                'connection_strength_multiplier': 1.5,
                'node_spacing': 100
            }
        }
        
        return game_data

    def _prepare_puzzle_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare puzzle nodes with enhanced properties"""
        prepared_nodes = []
        
        for node in nodes:
            prepared_node = node.copy()
            # Add visual properties
            prepared_node['visual'] = {
                'background_color': self._get_element_color(node['type']),
                'border_color': '#333333',
                'text_color': '#FFFFFF',
                'size': {
                    'width': len(node['content']) * 5 + 50,
                    'height': 60
                }
            }
            # Add interaction properties
            prepared_node['interaction'] = {
                'draggable': True,
                'connectable': True,
                'snap_points': self._generate_snap_points(node['position'])
            }
            prepared_nodes.append(prepared_node)
        
        return prepared_nodes

    def _generate_snap_points(self, position: Dict[str, int]) -> List[Dict[str, int]]:
        """Generate snap points for puzzle nodes"""
        return [
            {'x': position['x'], 'y': position['y'] - 30},  # Top
            {'x': position['x'] + 30, 'y': position['y']},  # Right
            {'x': position['x'], 'y': position['y'] + 30},  # Bottom
            {'x': position['x'] - 30, 'y': position['y']}   # Left
        ]
