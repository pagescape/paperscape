"""
Game state management service for the Research Paper Game Platform.
Handles game progress, achievements, and user interaction data.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Union

class GameState:
    def __init__(self):
        self.current_games: Dict[str, Dict] = {}
        self.user_progress: Dict[str, Dict] = {}
        self.achievements: Dict[str, List] = {}

    def create_game_session(self, game_id: str, game_type: str, content: Dict) -> str:
        """Create a new game session with initial state."""
        session_id = f"{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_games[session_id] = {
            "game_id": game_id,
            "type": game_type,
            "content": content,
            "state": {
                "current_stage": 0,
                "points": 0,
                "completed_tasks": [],
                "discoveries": [],
                "start_time": datetime.now().isoformat()
            }
        }
        return session_id

    def update_game_state(self, session_id: str, updates: Dict) -> bool:
        """Update the state of an ongoing game session."""
        if session_id not in self.current_games:
            return False
        
        self.current_games[session_id]["state"].update(updates)
        return True

    def get_game_state(self, session_id: str) -> Optional[Dict]:
        """Retrieve the current state of a game session."""
        return self.current_games.get(session_id)

    def award_achievement(self, user_id: str, achievement: Dict) -> None:
        """Award an achievement to a user."""
        if user_id not in self.achievements:
            self.achievements[user_id] = []
        
        achievement["timestamp"] = datetime.now().isoformat()
        self.achievements[user_id].append(achievement)

    def update_progress(self, user_id: str, game_id: str, progress: Dict) -> None:
        """Update user's progress in a specific game."""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        
        if game_id not in self.user_progress[user_id]:
            self.user_progress[user_id][game_id] = {
                "started_at": datetime.now().isoformat(),
                "stages_completed": [],
                "total_points": 0,
                "achievements": []
            }
        
        self.user_progress[user_id][game_id].update(progress)

    def get_user_progress(self, user_id: str, game_id: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """Get user's progress for all games or a specific game."""
        if user_id not in self.user_progress:
            return {} if game_id else []
        
        if game_id:
            return self.user_progress[user_id].get(game_id, {})
        return self.user_progress[user_id]

    def end_game_session(self, session_id: str) -> Optional[Dict]:
        """End a game session and return final state."""
        if session_id not in self.current_games:
            return None
        
        game_data = self.current_games[session_id]
        game_data["state"]["end_time"] = datetime.now().isoformat()
        
        # Calculate final score and achievements
        final_state = self._calculate_final_state(game_data)
        
        # Remove from active sessions
        del self.current_games[session_id]
        
        return final_state

    def _calculate_final_state(self, game_data: Dict) -> Dict:
        """Calculate final state including score and achievements."""
        state = game_data["state"]
        game_type = game_data["type"]
        
        # Calculate completion percentage
        total_tasks = len(game_data["content"].get("tasks", []))
        completed_tasks = len(state["completed_tasks"])
        completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate time spent
        start_time = datetime.fromisoformat(state["start_time"])
        end_time = datetime.fromisoformat(state["end_time"])
        time_spent = (end_time - start_time).total_seconds()
        
        return {
            "game_id": game_data["game_id"],
            "type": game_type,
            "final_score": state["points"],
            "completion_percentage": completion_percentage,
            "time_spent_seconds": time_spent,
            "discoveries": state["discoveries"],
            "achievements": self._generate_achievements(state, game_type)
        }

    def _generate_achievements(self, state: Dict, game_type: str) -> List[Dict]:
        """Generate achievements based on game performance."""
        achievements = []
        
        # Speed achievement
        if state.get("time_spent_seconds", float("inf")) < 300:  # 5 minutes
            achievements.append({
                "id": "speed_demon",
                "name": "Speed Demon",
                "description": "Completed the game in under 5 minutes!"
            })
        
        # Points achievement
        if state.get("points", 0) >= 1000:
            achievements.append({
                "id": "point_master",
                "name": "Point Master",
                "description": "Scored over 1000 points!"
            })
        
        # Discovery achievement
        if len(state.get("discoveries", [])) >= 5:
            achievements.append({
                "id": "explorer",
                "name": "Explorer",
                "description": "Made 5 or more discoveries!"
            })
        
        return achievements

# Global game state manager instance
game_state_manager = GameState()
