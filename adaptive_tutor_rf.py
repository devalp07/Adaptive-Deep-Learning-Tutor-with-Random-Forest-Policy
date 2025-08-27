"""
Adaptive Deep Learning Tutor with Random Forest Policy
-----------------------------------------------------
Enhanced version that replaces the neural contextual bandit with a Random Forest policy
that estimates question effectiveness based on user state and question features.

Features:
- Semantic similarity via Sentence-BERT (SBERT)
- Random Forest policy for question recommendation (like DQN but more interpretable)
- Automatic embeddings & difficulty predictor
- Streamlit-based web interface
- Enhanced reward system and state representation
- Timer for each question
- Fixed question persistence
- Detailed session review with performance analysis

Requirements:
pip install sentence-transformers torch sklearn pandas numpy matplotlib streamlit scipy

Author: Generated for user with 200 questions dataset
"""

import os
import time
import math
import random
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import beta

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# Streamlit for web interface
import streamlit as st

# -------------------------------
# Configuration
# -------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast SBERT
EMBEDDING_DIM = 384
RF_N_ESTIMATORS = 100  # Number of trees in Random Forest
RF_MAX_DEPTH = 10      # Maximum depth of trees
REPLAY_CAPACITY = 2000
BATCH_SIZE = 32
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_QUESTIONS_PER_SESSION = 50
SESSION_EXPIRY_HOURS = 24
TIME_LIMIT_PER_QUESTION = 120  # 2 minutes per question

# -------------------------------
# Utilities
# -------------------------------

def load_questions(csv_path: str) -> pd.DataFrame:
    """Load questions from CSV file with validation"""
    df = pd.read_csv(csv_path)
    required_cols = [
        "ID","Question","Topic","Difficulty",
        "Option_A","Option_B","Option_C","Option_D",
        "Correct_Answer","Solution_Steps","Video_Link"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df.reset_index(drop=True)
    df["Difficulty"] = pd.to_numeric(df["Difficulty"], errors="coerce").fillna(1.0)
    df["ID"] = df["ID"].astype(str)
    return df

# -------------------------------
# Embeddings & Difficulty Predictor
# -------------------------------
class EmbeddingStore:
    """Handles text embeddings using Sentence-BERT"""
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        st.info(f"Loading SBERT model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

    def embed_questions(self, df: pd.DataFrame, text_fields=("Question","Solution_Steps")) -> np.ndarray:
        """Generate embeddings for all questions"""
        corpus = (df[text_fields[0]].fillna("") + " \n " + df[text_fields[1]].fillna("")).tolist()
        embeddings = self.model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

# Difficulty predictor (simple MLP using embeddings)
class DifficultyPredictor:
    """Predicts question difficulty based on embeddings"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, early_stopping=True)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the difficulty predictor"""
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict difficulty for given embeddings"""
        if not self.fitted:
            raise RuntimeError("Predictor not trained")
        return self.model.predict(self.scaler.transform(X))

# -------------------------------
# User State
# -------------------------------
class UserState:
    """Tracks user learning state and history"""
    def __init__(self, username: str):
        self.username = username
        self.skill = 0.0  # Initial skill level
        self.history: List[Dict[str,Any]] = []  # Question history
        self.asked_ids = set()  # IDs of questions already asked
        self.topic_scores = {}  # Performance by topic
        self.cur_q = None  # Current question
        self.start_time = None  # Start time of current question
        self.hint_views = 0  # Number of hints viewed for current question
        self.theta_history = []  # History of skill levels
        self.diff_history = []  # History of question difficulties
        self.session_start = datetime.datetime.now()  # Session start time
        self.questions_attempted = 0  # Total questions attempted
        self.time_expired = False  # Flag for time expiration
        self.rewards = 0  # Total rewards earned
        self.consecutive_correct = 0  # Consecutive correct answers
        self.learning_path = []  # Personalized learning path
        self.current_question_index = -1  # Track current question in learning path
        self.answer_submitted = False  # Track if answer has been submitted
        self.session_review_data = []  # Store data for session review

    def update_from_result(self, q, correct: bool, time_sec: float, sim: float, selected_answer: str):
        """Update user state based on question result"""
        topic = q['Topic']
        c, tot = self.topic_scores.get(topic, (0,0))
        self.topic_scores[topic] = (c + (1 if correct else 0), tot + 1)
        self.asked_ids.add(str(q['ID']))
        self.questions_attempted += 1

        # Store detailed history
        self.history.append({
            'id': q['ID'], 'topic': topic, 'diff': float(q['Difficulty']),
            'correct': correct, 'time_sec': time_sec,
            'hints': self.hint_views, 'sim': sim
        })

        # Store data for session review
        self.session_review_data.append({
            'id': q['ID'],
            'question': q['Question'],
            'topic': topic,
            'difficulty': float(q['Difficulty']),
            'options': {
                'A': q['Option_A'],
                'B': q['Option_B'],
                'C': q['Option_C'],
                'D': q['Option_D']
            },
            'correct_answer': q['Correct_Answer'],
            'selected_answer': selected_answer,
            'is_correct': correct,
            'time_taken': time_sec,
            'hints_used': self.hint_views,
            'solution': q['Solution_Steps'],
            'video_link': q['Video_Link']
        })

        # Update skill using logistic model
        self.skill = update_skill(self.skill, float(q['Difficulty']), correct, time_sec, self.hint_views)
        self.theta_history.append(self.skill)
        self.diff_history.append(float(q['Difficulty']))

        # Update consecutive correct count
        if correct:
            self.consecutive_correct += 1
            self.rewards += 1
        else:
            self.consecutive_correct = 0
            self.rewards -= 0.5

# -------------------------------
# Random Forest Policy (Replaces Neural Network)
# -------------------------------
class RandomForestPolicy:
    """
    Random Forest based policy that estimates question effectiveness
    Similar to DQN but more interpretable and stable for smaller datasets
    """
    def __init__(self, n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.fitted = False
        self.replay = []  # Experience replay buffer
        self.capacity = REPLAY_CAPACITY
        self.epsilon = EPSILON_START

    def score(self, state_vec: np.ndarray, qfeat_vec: np.ndarray) -> float:
        """Score a question based on user state and question features"""
        if not self.fitted or random.random() < self.epsilon:
            return random.random()  # Exploration

        # Combine state and question features
        combined_features = np.concatenate([state_vec, qfeat_vec])
        return self.model.predict(combined_features.reshape(1, -1))[0]

    def add_experience(self, state_vec, qfeat_vec, reward):
        """Add experience to replay buffer"""
        if len(self.replay) >= self.capacity:
            self.replay.pop(0)
        self.replay.append((state_vec, qfeat_vec, reward))

    def train_from_replay(self):
        """Train the Random Forest on experiences from replay buffer"""
        if len(self.replay) < min(BATCH_SIZE, 20):
            return

        # Prepare training data
        states, qfeats, rewards = zip(*self.replay)
        X = [np.concatenate([s, q]) for s, q in zip(states, qfeats)]
        y = rewards

        # Train the model
        self.model.fit(X, y)
        self.fitted = True

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        """Save the trained model"""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        """Load a trained model"""
        import joblib
        self.model = joblib.load(path)
        self.fitted = True

# -------------------------------
# Helper: Build features for policy
# -------------------------------
class FeatureBuilder:
    """Builds feature vectors for user state and questions"""
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.emb = embeddings  # shape (N, D)

        # One-hot encoder for topic
        self.topics = np.array(self.df['Topic'].fillna('NA'))
        self.topic_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.topic_encoder.fit(self.topics.reshape(-1,1))

    def question_feature(self, idx: int) -> np.ndarray:
        """Create feature vector for a question"""
        # Embedding (first 64 dimensions for efficiency)
        emb = self.emb[idx]
        emb_small = emb[:64]

        # Difficulty
        diff = np.array([self.df.loc[idx,'Difficulty']], dtype=float)

        # Topic one-hot encoding
        topic_ohe = self.topic_encoder.transform([[self.df.loc[idx,'Topic']]])[0]

        # Combine all features
        feat = np.concatenate([emb_small, diff, topic_ohe], axis=0)
        return feat

    def state_vector(self, user_state: UserState) -> np.ndarray:
        """Create feature vector for user state"""
        # Basic metrics
        skill = np.array([user_state.skill], dtype=float)
        consec = np.array([user_state.consecutive_correct], dtype=float)

        # Time metrics
        if user_state.history:
            recent_times = [h['time_sec'] for h in user_state.history[-5:]]
            avg_time = np.array([np.mean(recent_times)])
            time_std = np.array([np.std(recent_times) if len(recent_times) > 1 else 0.0])
        else:
            avg_time = np.array([0.0])
            time_std = np.array([0.0])

        # Topic performance
        topic_acc = []
        topic_count = []
        for t in self.topic_encoder.categories_[0]:
            c, tot = user_state.topic_scores.get(t, (0,0))
            topic_acc.append(c/tot if tot>0 else 0.0)
            topic_count.append(tot)

        topic_acc = np.array(topic_acc, dtype=float)
        topic_count = np.array(topic_count, dtype=float)

        # Recent performance
        if len(user_state.history) >= 3:
            last_3 = [1 if h['correct'] else 0 for h in user_state.history[-3:]]
            recent_accuracy = np.array([np.mean(last_3)])
        else:
            recent_accuracy = np.array([0.0])

        # Combine all state features
        state = np.concatenate([
            skill, consec, avg_time, time_std,
            topic_acc, topic_count, recent_accuracy
        ], axis=0)
        return state

# -------------------------------
# Skill update model
# -------------------------------
def probability_correct(theta: float, diff: float) -> float:
    """Calculate probability of correct answer given skill and difficulty"""
    return 1.0 / (1.0 + math.exp(-(theta - diff)))

def update_skill(theta: float, diff: float, correct: bool, time_sec: float, hint_views: int) -> float:
    """Update user skill based on performance"""
    K = 0.6  # Learning rate
    p = probability_correct(theta, diff)
    y = 1.0 if correct else 0.0

    # Penalties for time and hints
    time_penalty = min(time_sec / TIME_LIMIT_PER_QUESTION, 1.5)
    hint_penalty = min(hint_views * 0.15, 0.6)
    penalty = max(0.4, 1.0 - 0.4*time_penalty - hint_penalty)

    # Update skill
    new_theta = theta + K * (y - p) * penalty
    return float(np.clip(new_theta, -3.0, 3.0))

# -------------------------------
# Reward calculation
# -------------------------------
def calculate_reward(correct: bool, time_taken: float, sim: float,
                    hint_views: int, difficulty: float, user_skill: float) -> float:
    """Calculate reward for a question attempt"""
    # Base reward
    base_reward = 1.0 if correct else -0.5

    # Time efficiency
    time_norm = time_taken / TIME_LIMIT_PER_QUESTION
    time_reward = 0.5 * (1.0 - time_norm) if correct else 0

    # Reasoning alignment
    reasoning_bonus = 0.7 * sim if sim >= 0.4 else 0

    # Hint penalty
    hint_penalty = -0.2 * hint_views

    # Difficulty scaling (more reward for harder questions)
    diff_bonus = 0.3 * (difficulty - user_skill) if correct and difficulty > user_skill else 0

    # Total reward with clamping
    total_reward = base_reward + time_reward + reasoning_bonus + hint_penalty + diff_bonus
    return max(-1.0, min(2.0, total_reward))

# -------------------------------
# Question selection strategy
# -------------------------------
def select_questions(pool_idx, user, fb, policy, df, sample_size=50):
    """Strategic question selection based on user weaknesses"""
    if not pool_idx:
        return []

    # Calculate topic performance
    topic_performance = {}
    for topic in df['Topic'].unique():
        c, tot = user.topic_scores.get(topic, (0,0))
        topic_performance[topic] = c/tot if tot > 0 else 0

    # Weight questions by topic weakness and difficulty appropriateness
    weights = []
    for idx in pool_idx:
        topic = df.loc[idx, 'Topic']
        weight = 1.0 - topic_performance.get(topic, 0)  # Higher weight for weaker topics

        # Adjust for difficulty (closer to user skill gets higher weight)
        diff = df.loc[idx, 'Difficulty']
        diff_weight = 1.0 / (1.0 + abs(user.skill - diff))

        weights.append(weight * diff_weight)

    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(pool_idx)) / len(pool_idx)

    # Sample strategically
    if len(pool_idx) <= sample_size:
        return pool_idx
    else:
        return np.random.choice(pool_idx, size=sample_size, replace=False, p=weights)

# -------------------------------
# Learning path generation
# -------------------------------
def generate_learning_path(user, df, fb, policy, num_questions=10):
    """Generate a personalized learning path for the user"""
    pool_idx = [i for i,qid in enumerate(df['ID']) if qid not in user.asked_ids]
    if not pool_idx:
        return []

    state_vec = fb.state_vector(user)
    question_scores = []

    # Score all available questions
    for idx in pool_idx:
        qf = fb.question_feature(idx)
        score = policy.score(state_vec, qf)
        question_scores.append((score, idx))

    # Sort by predicted effectiveness
    question_scores.sort(reverse=True)

    # Select top questions for learning path
    learning_path = [idx for _, idx in question_scores[:num_questions]]
    return learning_path

# -------------------------------
# Timer component
# -------------------------------
def timer_component(start_time, time_limit):
    """Display a timer for the current question"""
    elapsed_time = time.time() - start_time
    remaining_time = max(0, time_limit - elapsed_time)

    # Format time as MM:SS
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)

    # Create progress bar
    progress = elapsed_time / time_limit
    st.progress(min(1.0, progress))

    # Display time
    st.write(f"⏰ Time remaining: {minutes:02d}:{seconds:02d}")

    # Check if time has expired
    if remaining_time <= 0:
        st.error("Time's up! Please submit your answer or skip to the next question.")
        return True

    return False

# -------------------------------
# Session Review Component
# -------------------------------
def show_session_review(user, df):
    """Display a comprehensive review of the session"""
    st.title("📊 Session Review")
    st.markdown(f"### Performance Summary for {user.username}")

    # Calculate overall statistics
    total_questions = len(user.session_review_data)
    correct_answers = sum(1 for q in user.session_review_data if q['is_correct'])
    accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    avg_time = np.mean([q['time_taken'] for q in user.session_review_data]) if total_questions > 0 else 0
    total_hints = sum(q['hints_used'] for q in user.session_review_data)

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Correct Answers", f"{correct_answers} ({accuracy:.1f}%)")
    with col3:
        st.metric("Average Time", f"{avg_time:.1f}s")
    with col4:
        st.metric("Hints Used", total_hints)

    # Display skill progression
    st.subheader("Skill Progression")
    if user.theta_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        qs = range(1, len(user.theta_history)+1)
        ax.plot(qs, user.theta_history, 'b-', label='Your Skill', linewidth=2)
        ax.plot(qs, user.diff_history, 'r--', label='Question Difficulty', linewidth=2)
        ax.fill_between(qs, user.theta_history, user.diff_history,
                       where=[th >= dh for th, dh in zip(user.theta_history, user.diff_history)],
                       alpha=0.3, color='green', label='Mastery Zone')
        ax.fill_between(qs, user.theta_history, user.diff_history,
                       where=[th < dh for th, dh in zip(user.theta_history, user.diff_history)],
                       alpha=0.3, color='red', label='Learning Zone')
        ax.set_xlabel('Question #')
        ax.set_ylabel('Skill/Difficulty')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Display topic-wise performance
    st.subheader("Topic-wise Performance")
    topic_stats = {}
    for q in user.session_review_data:
        topic = q['topic']
        if topic not in topic_stats:
            topic_stats[topic] = {'total': 0, 'correct': 0}
        topic_stats[topic]['total'] += 1
        if q['is_correct']:
            topic_stats[topic]['correct'] += 1

    if topic_stats:
        topics = list(topic_stats.keys())
        accuracies = [(topic_stats[t]['correct'] / topic_stats[t]['total'] * 100) for t in topics]
        counts = [topic_stats[t]['total'] for t in topics]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(topics))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='skyblue')
        bars2 = ax.bar(x + width/2, counts, width, label='Question Count', color='lightcoral')

        ax.set_xlabel('Topics')
        ax.set_title('Performance by Topic')
        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend()

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)

    # Display detailed question review
    st.subheader("Detailed Question Review")
    for i, q_data in enumerate(user.session_review_data):
        with st.expander(f"Question {i+1}: {q_data['question'][:50]}... ({'Correct' if q_data['is_correct'] else 'Incorrect'})"):
            st.markdown(f"**Topic:** {q_data['topic']} | **Difficulty:** {q_data['difficulty']:.1f}")
            st.markdown(f"**Time Taken:** {q_data['time_taken']:.1f}s | **Hints Used:** {q_data['hints_used']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Options:**")
                st.write(f"A) {q_data['options']['A']}")
                st.write(f"B) {q_data['options']['B']}")
                st.write(f"C) {q_data['options']['C']}")
                st.write(f"D) {q_data['options']['D']}")

            with col2:
                st.markdown("**Your Answer:**")
                if q_data['is_correct']:
                    st.success(f"{q_data['selected_answer']} ✓ (Correct)")
                else:
                    st.error(f"{q_data['selected_answer']} ✗ (Incorrect)")

                st.markdown("**Correct Answer:**")
                st.info(f"{q_data['correct_answer']} ✓")

            st.markdown("**Solution:**")
            st.write(q_data['solution'])

            if q_data['video_link'] and str(q_data['video_link']).lower() != 'nan':
                st.video(q_data['video_link'])

    # Download session report
    st.subheader("Download Session Report")
    report_data = {
        "username": user.username,
        "session_date": user.session_start.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "average_time": avg_time,
        "total_hints": total_hints,
        "final_skill": user.skill,
        "questions": user.session_review_data
    }

    report_json = json.dumps(report_data, indent=2)
    st.download_button(
        label="Download Session Report (JSON)",
        data=report_json,
        file_name=f"session_report_{user.username}_{user.session_start.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# -------------------------------
# Streamlit Web Interface
# -------------------------------
def run_streamlit_app(csv_path: str):
    """Run the adaptive tutor as a Streamlit web application"""
    st.set_page_config(
        page_title="Adaptive Deep Learning Tutor",
        page_icon="📚",
        layout="wide"
    )

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'df' not in st.session_state:
        try:
            st.session_state.df = load_questions(csv_path)
        except Exception as e:
            st.error(f"Error loading questions: {e}")
            return
    if 'fb' not in st.session_state:
        with st.spinner("Loading AI models..."):
            try:
                emb_store = EmbeddingStore()
                emb = emb_store.embed_questions(st.session_state.df)
                st.session_state.fb = FeatureBuilder(st.session_state.df, emb)
            except Exception as e:
                st.error(f"Error initializing models: {e}")
                return
    if 'policy' not in st.session_state:
        try:
            state_dim = 3 + st.session_state.fb.topic_encoder.categories_[0].shape[0]
            qfeat_dim = 64 + 1 + st.session_state.fb.topic_encoder.categories_[0].shape[0]
            st.session_state.policy = RandomForestPolicy()
        except Exception as e:
            st.error(f"Error initializing policy: {e}")
            return
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'answer_submitted' not in st.session_state:
        st.session_state.answer_submitted = False
    if 'selected_answer' not in st.session_state:
        st.session_state.selected_answer = None
    if 'hint_shown' not in st.session_state:
        st.session_state.hint_shown = False
    if 'show_review' not in st.session_state:
        st.session_state.show_review = False

    # Show session review if enabled
    if st.session_state.show_review and st.session_state.user:
        show_session_review(st.session_state.user, st.session_state.df)
        if st.button("Start New Session"):
            st.session_state.show_review = False
            st.session_state.user = None
            st.session_state.current_question = None
            st.session_state.answer_submitted = False
            st.session_state.selected_answer = None
            st.session_state.hint_shown = False
            st.rerun()
        return

    # Main application
    st.title("📚 Adaptive Deep Learning Tutor")
    st.markdown("""
    This adaptive tutoring system personalizes your learning experience by:
    - Assessing your skill level in real-time
    - Selecting questions that match your current ability
    - Adjusting difficulty based on your performance
    - Providing detailed feedback and explanations
    """)

    # Sidebar for user information
    with st.sidebar:
        st.header("User Profile")
        if st.session_state.user is None:
            username = st.text_input("Enter your name:", "Student")
            if st.button("Start Session"):
                st.session_state.user = UserState(username)
                st.session_state.answer_submitted = False
                st.session_state.selected_answer = None
                st.session_state.hint_shown = False
                st.session_state.show_review = False
                st.rerun()
        else:
            st.write(f"**User:** {st.session_state.user.username}")
            st.write(f"**Skill Level:** {st.session_state.user.skill:.2f}")
            st.write(f"**Questions Attempted:** {st.session_state.user.questions_attempted}")
            st.write(f"**Correct Streak:** {st.session_state.user.consecutive_correct}")

            if st.button("End Session"):
                if st.session_state.user.history:
                    st.session_state.show_review = True
                st.rerun()

    # Main content area
    if st.session_state.user is None:
        st.info("Please enter your name and start a session from the sidebar.")
        return

    user = st.session_state.user
    df = st.session_state.df
    fb = st.session_state.fb
    policy = st.session_state.policy

    # Check session limits
    if (datetime.datetime.now() - user.session_start).total_seconds() > SESSION_EXPIRY_HOURS * 3600:
        st.error("Session expired. Please start a new session.")
        if user.history:
            st.session_state.show_review = True
        st.rerun()

    if user.questions_attempted >= MAX_QUESTIONS_PER_SESSION:
        st.success("Congratulations! You've completed today's session.")
        if user.history:
            st.session_state.show_review = True
        st.rerun()

    # Generate or get next question if needed
    if not user.learning_path or st.session_state.answer_submitted:
        user.learning_path = generate_learning_path(user, df, fb, policy)
        user.current_question_index = -1
        st.session_state.answer_submitted = False
        st.session_state.selected_answer = None
        st.session_state.hint_shown = False

    if not user.learning_path:
        st.info("You've attempted all available questions!")
        if user.history:
            st.session_state.show_review = True
        st.rerun()

    # Get next question from learning path
    if user.current_question_index < 0 or st.session_state.answer_submitted:
        user.current_question_index = (user.current_question_index + 1) % len(user.learning_path)
        next_idx = user.learning_path[user.current_question_index]
        st.session_state.current_question = df.loc[next_idx]
        user.cur_q = st.session_state.current_question
        user.start_time = time.time()
        user.hint_views = 0
        st.session_state.answer_submitted = False
        st.session_state.selected_answer = None
        st.session_state.hint_shown = False
        st.rerun()

    q = st.session_state.current_question

    # Display question
    st.header(f"Question {q['ID']}")
    st.subheader(q['Question'])

    # Display timer
    time_expired = timer_component(user.start_time, TIME_LIMIT_PER_QUESTION)
    if time_expired:
        user.time_expired = True

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**A)** {q['Option_A']}")
        st.write(f"**B)** {q['Option_B']}")
    with col2:
        st.write(f"**C)** {q['Option_C']}")
        st.write(f"**D)** {q['Option_D']}")

    st.caption(f"Topic: {q['Topic']} | Difficulty: {q['Difficulty']:.1f}")

    # User interaction - only show if answer not submitted
    if not st.session_state.answer_submitted and not user.time_expired:
        # Use a key to ensure the radio button doesn't reset
        selected_answer = st.radio(
            "Select your answer:",
            ["A", "B", "C", "D"],
            horizontal=True,
            key=f"answer_{q['ID']}"
        )
        st.session_state.selected_answer = selected_answer

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Submit Answer", key=f"submit_{q['ID']}"):
                if st.session_state.selected_answer:
                    time_taken = time.time() - user.start_time
                    correct = (st.session_state.selected_answer == str(q['Correct_Answer']).strip())

                    # Get reasoning if provided
                    reason = st.text_area("Optional: Explain your reasoning", key=f"reason_{q['ID']}")
                    sim = 0.0
                    if reason.strip():
                        emb_store = EmbeddingStore()
                        emb_user = emb_store.model.encode([reason], convert_to_numpy=True)
                        emb_sol = emb_store.model.encode([str(q['Solution_Steps'])], convert_to_numpy=True)
                        sim = float(cosine_similarity(emb_user, emb_sol)[0][0])

                    # Update user state
                    user.update_from_result(q, correct, time_taken, sim, st.session_state.selected_answer)

                    # Calculate reward and update policy
                    reward = calculate_reward(correct, time_taken, sim, user.hint_views,
                                            q['Difficulty'], user.skill)
                    state_vec = fb.state_vector(user)
                    policy.add_experience(state_vec, fb.question_feature(user.learning_path[user.current_question_index]), reward)
                    policy.train_from_replay()

                    # Show feedback
                    if correct:
                        st.success("✅ Correct! Well done.")
                    else:
                        st.error(f"❌ Incorrect. The correct answer is {q['Correct_Answer']}.")

                    st.subheader("Solution Explanation")
                    st.write(q['Solution_Steps'])

                    if q['Video_Link'] and str(q['Video_Link']).lower() != 'nan':
                        st.video(q['Video_Link'])

                    # Save policy periodically
                    if user.questions_attempted % 10 == 0:
                        policy.save(f"{user.username}_policy.pkl")

                    st.session_state.answer_submitted = True
                    st.rerun()
                else:
                    st.warning("Please select an answer before submitting.")

        with col2:
            if st.button("💡 Get Hint", key=f"hint_{q['ID']}"):
                user.hint_views += 1
                st.session_state.hint_shown = True
                sol = str(q['Solution_Steps'])
                words = sol.split()[:10]  # First 10 words as hint
                st.info("Hint: " + " ".join(words))

        with col3:
            if st.button("⏭️ Skip Question", key=f"skip_{q['ID']}"):
                # Small penalty for skipping
                reward = -0.2
                state_vec = fb.state_vector(user)
                policy.add_experience(state_vec, fb.question_feature(user.learning_path[user.current_question_index]), reward)
                policy.train_from_replay()

                # Record the skip in session review
                user.session_review_data.append({
                    'id': q['ID'],
                    'question': q['Question'],
                    'topic': q['Topic'],
                    'difficulty': float(q['Difficulty']),
                    'options': {
                        'A': q['Option_A'],
                        'B': q['Option_B'],
                        'C': q['Option_C'],
                        'D': q['Option_D']
                    },
                    'correct_answer': q['Correct_Answer'],
                    'selected_answer': 'Skipped',
                    'is_correct': False,
                    'time_taken': time.time() - user.start_time,
                    'hints_used': user.hint_views,
                    'solution': q['Solution_Steps'],
                    'video_link': q['Video_Link']
                })

                st.session_state.answer_submitted = True
                st.rerun()

    # Show hint if it was previously shown
    elif st.session_state.hint_shown:
        sol = str(q['Solution_Steps'])
        words = sol.split()[:10]
        st.info("Hint: " + " ".join(words))

    # Progress visualization
    if user.history:
        st.subheader("Your Progress")
        fig, ax = plt.subplots(figsize=(10, 4))
        qs = range(1, len(user.theta_history)+1)
        ax.plot(qs, user.theta_history, 'b-', label='Your Skill', linewidth=2)
        ax.plot(qs, user.diff_history, 'r--', label='Question Difficulty', linewidth=2)
        ax.fill_between(qs, user.theta_history, user.diff_history,
                       where=[th >= dh for th, dh in zip(user.theta_history, user.diff_history)],
                       alpha=0.3, color='green', label='Mastery Zone')
        ax.fill_between(qs, user.theta_history, user.diff_history,
                       where=[th < dh for th, dh in zip(user.theta_history, user.diff_history)],
                       alpha=0.3, color='red', label='Learning Zone')
        ax.set_xlabel('Question #')
        ax.set_ylabel('Skill/Difficulty')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# -------------------------------
# Main execution
# -------------------------------
if __name__ == '__main__':
    csv_path = "Aptitude_questions.csv"  # Your CSV file path

    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        st.info("Please make sure your questions CSV file is named 'Aptitude_questions.csv' and placed in the same directory.")
    else:
        run_streamlit_app(csv_path)
