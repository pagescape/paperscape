import os
import json
import spacy
import PyPDF2
import numpy as np
from gensim.summarization import summarize
from transformers import pipeline
from pdfminer.high_level import extract_text

class PaperProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            # If model not found, download it
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.key_phrases = []
        self.summary = ""
        self.concepts = []

    def process_paper(self, file_path):
        """Process the uploaded research paper"""
        try:
            # Extract text from the file
            text = self._extract_text(file_path)
            if not text.strip():
                raise ValueError("No text could be extracted from the file")

            # Process the text
            self.summary = self._generate_summary(text)
            self.key_phrases = self._extract_key_phrases(text)
            self.concepts = self._identify_concepts(text)
            
            # Prepare and return the processed data
            return {
                'summary': self.summary,
                'key_phrases': self.key_phrases,
                'concepts': self.concepts
            }
            
        except Exception as e:
            raise Exception(f"Error processing paper: {str(e)}")

    def _extract_text(self, file_path):
        """Extract text from PDF or text file"""
        try:
            if file_path.lower().endswith('.pdf'):
                try:
                    # Try pdfminer first
                    text = extract_text(file_path)
                    if text.strip():
                        return text
                except:
                    # Fallback to PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
        except Exception as e:
            raise Exception(f"Error extracting text from file: {str(e)}")

    def _generate_summary(self, text):
        """Generate a concise summary of the paper"""
        try:
            # Clean and prepare text
            text = text.replace('\n', ' ').strip()
            
            # Split text into chunks of 1024 characters
            chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
            
            # Generate summary for each chunk
            summaries = []
            for chunk in chunks[:3]:  # Process first 3 chunks only
                if chunk.strip():
                    try:
                        summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                        summaries.append(summary[0]['summary_text'])
                    except:
                        continue
            
            if summaries:
                return ' '.join(summaries)
            else:
                return text[:500] + "..."  # Fallback to first 500 chars
                
        except Exception as e:
            print(f"Error generating summary: {e}")
            return text[:500] + "..."  # Fallback to first 500 chars

    def _extract_key_phrases(self, text):
        """Extract key phrases from the text"""
        try:
            # Process with spaCy
            doc = self.nlp(text[:10000])  # Process first 10000 chars only
            
            # Extract noun phrases and named entities
            phrases = []
            
            # Get noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit to 4 words
                    phrases.append(chunk.text.strip())
            
            # Get named entities
            for ent in doc.ents:
                if len(ent.text.split()) <= 4:  # Limit to 4 words
                    phrases.append(ent.text.strip())
            
            # Remove duplicates and sort by length
            phrases = list(set(phrases))
            phrases.sort(key=len, reverse=True)
            
            return phrases[:20]  # Return top 20 phrases
            
        except Exception as e:
            print(f"Error extracting key phrases: {e}")
            return []

    def _identify_concepts(self, text):
        """Identify main concepts from the text"""
        try:
            # Process with spaCy
            doc = self.nlp(text[:10000])  # Process first 10000 chars only
            
            # Extract concepts (nouns and their related words)
            concepts = {}
            
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    # Get the base form of the word
                    concept = token.lemma_.lower()
                    
                    if concept not in concepts:
                        concepts[concept] = {
                            'related_words': set(),
                            'count': 0
                        }
                    
                    # Increment count
                    concepts[concept]['count'] += 1
                    
                    # Add related words
                    for related in token.children:
                        if related.pos_ in ['ADJ', 'VERB']:
                            concepts[concept]['related_words'].add(related.lemma_.lower())
            
            # Convert to list and sort by count
            concept_list = []
            for concept, data in concepts.items():
                if data['count'] >= 2:  # Only include concepts that appear at least twice
                    concept_list.append({
                        'concept': concept,
                        'count': data['count'],
                        'related_words': list(data['related_words'])
                    })
            
            concept_list.sort(key=lambda x: x['count'], reverse=True)
            return concept_list[:15]  # Return top 15 concepts
            
        except Exception as e:
            print(f"Error identifying concepts: {e}")
            return []

    def _prepare_game_data(self):
        """Prepare data structure for game generation"""
        return {
            'summary': self.summary,
            'key_phrases': self.key_phrases,
            'concepts': self.concepts,
            'game_elements': {
                'quiz': self._generate_quiz_questions(),
                'simulation': self._prepare_simulation_data(),
                'puzzle': self._prepare_puzzle_data()
            }
        }

    def _generate_quiz_questions(self):
        """Generate quiz questions from the paper content"""
        questions = []
        for concept in self.concepts:
            if concept['concept'] == 'definition':
                questions.append({
                    'type': 'multiple_choice',
                    'question': f"What is the correct definition of {concept['concept']}?",
                    'correct_answer': concept['related_words'][0],
                    'options': [concept['related_words'][0]] + [c['related_words'][0] for c in self.concepts if c != concept][:3]
                })
            elif concept['concept'] == 'methodology':
                questions.append({
                    'type': 'true_false',
                    'question': f"Is the following statement about the methodology correct? {concept['related_words'][0]}",
                    'correct_answer': True,
                    'explanation': concept['related_words'][0]
                })
        return questions

    def _prepare_simulation_data(self):
        """Prepare data for interactive simulations"""
        simulation_data = {
            'type': 'particle_system',
            'elements': [],
            'interactions': []
        }
        
        # Create simulation elements from concepts
        for i, concept in enumerate(self.concepts):
            element = {
                'id': f"element_{i}",
                'type': concept['concept'],
                'label': concept['concept'],
                'description': ', '.join(concept['related_words']),
                'properties': {
                    'mass': 1 + i,
                    'charge': (-1 if i % 2 == 0 else 1),
                    'size': len(concept['related_words']) + 5
                }
            }
            simulation_data['elements'].append(element)
        
        # Create interactions between related concepts
        for i, concept1 in enumerate(self.concepts):
            for j, concept2 in enumerate(self.concepts[i+1:], i+1):
                if any(keyword in concept2['related_words'] for keyword in concept1['related_words']):
                    interaction = {
                        'source': f"element_{i}",
                        'target': f"element_{j}",
                        'strength': 0.5,
                        'type': 'attraction'
                    }
                    simulation_data['interactions'].append(interaction)
        
        return simulation_data

    def _prepare_puzzle_data(self):
        """Prepare data for drag-and-drop puzzles"""
        puzzle_data = {
            'type': 'concept_map',
            'nodes': [],
            'connections': []
        }
        
        # Create puzzle pieces from concepts
        for i, concept in enumerate(self.concepts):
            node = {
                'id': f"node_{i}",
                'type': concept['concept'],
                'content': ', '.join(concept['related_words']),
                'keywords': concept['related_words'],
                'position': {
                    'x': np.random.randint(100, 700),
                    'y': np.random.randint(100, 500)
                }
            }
            puzzle_data['nodes'].append(node)
        
        # Create connections between related concepts
        for i, concept1 in enumerate(self.concepts):
            for j, concept2 in enumerate(self.concepts[i+1:], i+1):
                shared_keywords = set(concept1['related_words']) & set(concept2['related_words'])
                if shared_keywords:
                    connection = {
                        'source': f"node_{i}",
                        'target': f"node_{j}",
                        'label': list(shared_keywords)[0],
                        'strength': len(shared_keywords)
                    }
                    puzzle_data['connections'].append(connection)
        
        return puzzle_data
