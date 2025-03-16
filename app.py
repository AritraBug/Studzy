# Studzy - AI Study Assistant
# A comprehensive study assistant to help students organize notes and create study plans

# --- Required Libraries ---
# pip install streamlit pandas numpy scikit-learn nltk python-dateutil matplotlib plotly

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import datetime
import json
import os
import random
from dateutil.parser import parse
import re
import time
import ssl
import certifi

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt', quiet=True)

# --- Helper Functions ---

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stop words, and lemmatizing"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

def generate_study_plan(subject, topics, difficulty, duration_days):
    """Generate a study plan based on subject, topics, difficulty and duration"""
    # Calculate study sessions based on difficulty and duration
    if difficulty == "Easy":
        sessions_per_day = 1
    elif difficulty == "Medium":
        sessions_per_day = 2
    else:  # Hard
        sessions_per_day = 3
    
    total_sessions = sessions_per_day * duration_days
    
    # Distribute topics across sessions
    topic_list = [topic.strip() for topic in topics.split(',')]
    study_plan = []
    
    # Current date
    current_date = datetime.datetime.now()
    
    # Generate study plan
    for day in range(1, duration_days + 1):
        date = current_date + datetime.timedelta(days=day-1)
        date_str = date.strftime("%Y-%m-%d")
        
        for session in range(1, sessions_per_day + 1):
            topic_idx = (day - 1) * sessions_per_day + session - 1
            # Cycle through topics if needed
            actual_idx = topic_idx % len(topic_list)
            
            topic = topic_list[actual_idx]
            
            # Generate random session time
            if session == 1:
                time_str = "Morning"
            elif session == 2:
                time_str = "Afternoon"
            else:
                time_str = "Evening"
            
            study_plan.append({
                "day": day,
                "date": date_str,
                "session": session,
                "time": time_str,
                "topic": topic,
                "duration_minutes": 60,
                "completed": False
            })
    
    return study_plan

def find_similar_notes(query, notes_data):
    """Find similar notes based on query using TF-IDF and cosine similarity"""
    if not notes_data:
        return []
    
    # Extract text from notes
    notes_text = [note['content'] for note in notes_data]
    
    # Preprocess query and notes
    preprocessed_query = preprocess_text(query)
    preprocessed_notes = [preprocess_text(text) for text in notes_text]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_notes + [preprocessed_query])
    
    # Calculate cosine similarity
    query_vector = tfidf_matrix[-1]
    notes_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, notes_vectors).flatten()
    
    # Get top similar notes
    similar_notes_indices = similarities.argsort()[::-1]
    similar_notes = [(notes_data[idx], similarities[idx]) for idx in similar_notes_indices if similarities[idx] > 0.1]
    
    return similar_notes

def generate_quiz(notes_data, num_questions=5):
    """Generate a quiz based on notes"""
    if not notes_data or len(notes_data) < 2:
        return []
    
    questions = []
    
    for _ in range(min(num_questions, len(notes_data))):
        # Randomly select a note
        note = random.choice(notes_data)
        
        # Create a question based on the note
        content = note['content']
        title = note['title']
        
        # Extract key concepts
        words = content.split()
        key_concepts = [word for word in words if len(word) > 4 and word.isalpha()]
        
        if key_concepts:
            # Create a fill-in-the-blank question
            concept = random.choice(key_concepts)
            question_text = content.replace(concept, "________", 1)
            
            # Generate incorrect options
            options = [concept]
            other_notes = [n for n in notes_data if n['id'] != note['id']]
            
            for _ in range(3):
                if other_notes:
                    other_note = random.choice(other_notes)
                    other_words = other_note['content'].split()
                    other_key_concepts = [word for word in other_words if len(word) > 4 and word.isalpha()]
                    
                    if other_key_concepts:
                        other_concept = random.choice(other_key_concepts)
                        if other_concept not in options:
                            options.append(other_concept)
            
            # If we don't have enough options, add some random words
            while len(options) < 4:
                random_word = f"Option{len(options)}"
                options.append(random_word)
            
            # Shuffle options
            random.shuffle(options)
            
            # Create the question dictionary
            question = {
                "question": f"Fill in the blank: {question_text}",
                "options": options,
                "correct_answer": concept,
                "source_note": title
            }
            
            questions.append(question)
    
    return questions

def load_data():
    """Load data from JSON files or create them if they don't exist"""
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize empty data structures
    notes_data = []
    plans_data = []
    quiz_data = []
    
    # Files to manage
    files = {
        'data/notes.json': notes_data,
        'data/study_plans.json': plans_data,
        'data/quiz_history.json': quiz_data
    }
    
    # Process each file
    for file_path, data_list in files.items():
        # Create file with empty list if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)
        else:
            # Try to load existing data
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only parse if not empty
                        loaded_data = json.loads(content)
                        # Update the appropriate list reference
                        if file_path == 'data/notes.json':
                            notes_data = loaded_data
                        elif file_path == 'data/study_plans.json':
                            plans_data = loaded_data
                        else:  # quiz_history.json
                            quiz_data = loaded_data
            except Exception:
                # If any error occurs, ensure file has valid JSON
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    return notes_data, plans_data, quiz_data

def save_data(notes_data, plans_data, quiz_data):
    """Save data to JSON files"""
    # Save notes data
    with open('data/notes.json', 'w') as f:
        json.dump(notes_data, f)
    
    # Save study plans data
    with open('data/study_plans.json', 'w') as f:
        json.dump(plans_data, f)
    
    # Save quiz history data
    with open('data/quiz_history.json', 'w') as f:
        json.dump(quiz_data, f)

# --- Main Application ---

def main():
    # Page configuration
    st.set_page_config(
        page_title="Studzy - AI Study Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Load data
    notes_data, plans_data, quiz_data = load_data()
    
    # Application title
    st.title("📚 Studzy - AI Study Assistant")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Notes", "Study Plans", "Quiz", "Search", "Settings"])
    
    # Dashboard
    if page == "Dashboard":
        display_dashboard(notes_data, plans_data, quiz_data)
    
    # Notes
    elif page == "Notes":
        manage_notes(notes_data, plans_data, quiz_data)
    
    # Study Plans
    elif page == "Study Plans":
        manage_study_plans(notes_data, plans_data, quiz_data)
    
    # Quiz
    elif page == "Quiz":
        take_quiz(notes_data, plans_data, quiz_data)
    
    # Search
    elif page == "Search":
        search_notes(notes_data, plans_data, quiz_data)
    
    # Settings
    elif page == "Settings":
        settings(notes_data, plans_data, quiz_data)

def display_dashboard(notes_data, plans_data, quiz_data):
    """Display the dashboard with key metrics and upcoming tasks"""
    st.header("Dashboard")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Notes", len(notes_data))
    
    with col2:
        active_plans = [plan for plan in plans_data if not all(session['completed'] for session in plan['sessions'])]
        st.metric("Active Study Plans", len(active_plans))
    
    with col3:
        if quiz_data:
            avg_score = sum(quiz['score'] for quiz in quiz_data) / len(quiz_data)
            st.metric("Average Quiz Score", f"{avg_score:.1f}%")
        else:
            st.metric("Average Quiz Score", "N/A")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if not notes_data and not plans_data and not quiz_data:
        st.info("No recent activity. Start by adding notes or creating a study plan!")
    else:
        # Show recent notes
        if notes_data:
            st.write("Recent Notes:")
            recent_notes = sorted(notes_data, key=lambda x: x.get('created_at', ''), reverse=True)[:3]
            for note in recent_notes:
                st.write(f"- {note['title']} ({note.get('created_at', 'Unknown date')})")
        
        # Show upcoming study sessions
        if plans_data:
            st.write("Upcoming Study Sessions:")
            all_sessions = []
            for plan in plans_data:
                for session in plan['sessions']:
                    if not session['completed']:
                        session_info = {
                            'plan_name': plan['name'],
                            'date': session['date'],
                            'topic': session['topic'],
                            'time': session['time']
                        }
                        all_sessions.append(session_info)
            
            upcoming_sessions = sorted(all_sessions, key=lambda x: x['date'])[:5]
            for session in upcoming_sessions:
                st.write(f"- {session['plan_name']}: {session['topic']} - {session['date']} ({session['time']})")
    
    # Progress charts
    st.subheader("Study Progress")
    
    if plans_data:
        # Create a progress chart
        plan_names = [plan['name'] for plan in plans_data]
        completed_sessions = []
        total_sessions = []
        
        for plan in plans_data:
            completed = sum(1 for session in plan['sessions'] if session['completed'])
            total = len(plan['sessions'])
            
            completed_sessions.append(completed)
            total_sessions.append(total)
        
        progress_data = {
            'Plan': plan_names,
            'Completed': completed_sessions,
            'Total': total_sessions
        }
        
        progress_df = pd.DataFrame(progress_data)
        progress_df['Progress'] = (progress_df['Completed'] / progress_df['Total'] * 100).round(1)
        
        # Create a bar chart
        fig = px.bar(
            progress_df,
            x='Plan',
            y='Progress',
            labels={'Progress': 'Completion Rate (%)'},
            title='Study Plan Progress'
        )
        
        st.plotly_chart(fig)
    else:
        st.info("No study plans available. Create a plan to track your progress!")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Add New Note"):
            st.session_state.page = "Notes"
            st.rerun()
    
    with col2:
        if st.button("Create Study Plan"):
            st.session_state.page = "Study Plans"
            st.rerun()
    
    with col3:
        if st.button("Take a Quiz"):
            st.session_state.page = "Quiz"
            st.rerun()

def manage_notes(notes_data, plans_data, quiz_data):
    """Manage notes - add, edit, delete, and view notes"""
    st.header("Notes")
    
    # Tabs for different note actions
    tab1, tab2 = st.tabs(["View Notes", "Add Note"])
    
    # View Notes Tab
    with tab1:
        if not notes_data:
            st.info("No notes available. Add a note to get started!")
        else:
            # Filter notes by subject
            subjects = list(set(note.get('subject', 'General') for note in notes_data))
            selected_subject = st.selectbox("Filter by Subject", ["All"] + subjects)
            
            filtered_notes = notes_data
            if selected_subject != "All":
                filtered_notes = [note for note in notes_data if note.get('subject', 'General') == selected_subject]
            
            # Display notes
            for i, note in enumerate(filtered_notes):
                with st.expander(f"{note['title']} ({note.get('subject', 'General')})"):
                    st.write(f"**Created:** {note.get('created_at', 'Unknown')}")
                    st.write(note['content'])
                    
                    # Edit and Delete buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"Edit", key=f"edit_{i}"):
                            st.session_state.edit_note_id = note['id']
                            st.session_state.edit_note_title = note['title']
                            st.session_state.edit_note_content = note['content']
                            st.session_state.edit_note_subject = note.get('subject', 'General')
                            st.rerun()
                    
                    with col2:
                        if st.button(f"Delete", key=f"delete_{i}"):
                            notes_data.remove(note)
                            save_data(notes_data, plans_data, quiz_data)
                            st.success(f"Note '{note['title']}' deleted successfully!")
                            st.rerun()
            
            # Edit note form
            if 'edit_note_id' in st.session_state:
                st.subheader("Edit Note")
                
                edit_title = st.text_input("Title", st.session_state.edit_note_title)
                edit_subject = st.text_input("Subject", st.session_state.edit_note_subject)
                edit_content = st.text_area("Content", st.session_state.edit_note_content, height=300)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Save Changes"):
                        # Find the note and update it
                        for note in notes_data:
                            if note['id'] == st.session_state.edit_note_id:
                                note['title'] = edit_title
                                note['subject'] = edit_subject
                                note['content'] = edit_content
                                note['updated_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                break
                        
                        save_data(notes_data, plans_data, quiz_data)
                        st.success(f"Note '{edit_title}' updated successfully!")
                        
                        # Clear session state
                        del st.session_state.edit_note_id
                        del st.session_state.edit_note_title
                        del st.session_state.edit_note_content
                        del st.session_state.edit_note_subject
                        
                        st.rerun()
                
                with col2:
                    if st.button("Cancel"):
                        # Clear session state
                        del st.session_state.edit_note_id
                        del st.session_state.edit_note_title
                        del st.session_state.edit_note_content
                        del st.session_state.edit_note_subject
                        
                        st.rerun()
    
    # Add Note Tab
    with tab2:
        st.subheader("Add New Note")
        
        note_title = st.text_input("Title")
        note_subject = st.text_input("Subject", "General")
        note_content = st.text_area("Content", height=300)
        
        if st.button("Add Note"):
            if note_title and note_content:
                # Create a new note
                new_note = {
                    "id": str(time.time()),
                    "title": note_title,
                    "subject": note_subject,
                    "content": note_content,
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                notes_data.append(new_note)
                save_data(notes_data, plans_data, quiz_data)
                
                st.success(f"Note '{note_title}' added successfully!")
                # Clear the form
                st.rerun()
            else:
                st.error("Please provide both a title and content for the note.")

def manage_study_plans(notes_data, plans_data, quiz_data):
    """Manage study plans - create, edit, and view study plans"""
    st.header("Study Plans")
    
    # Tabs for different study plan actions
    tab1, tab2 = st.tabs(["View Plans", "Create Plan"])
    
    # View Plans Tab
    with tab1:
        if not plans_data:
            st.info("No study plans available. Create a plan to get started!")
        else:
            # Display plans
            for i, plan in enumerate(plans_data):
                with st.expander(f"{plan['name']} ({plan['subject']})"):
                    st.write(f"**Created:** {plan.get('created_at', 'Unknown')}")
                    st.write(f"**Duration:** {plan['duration_days']} days")
                    st.write(f"**Difficulty:** {plan['difficulty']}")
                    
                    # Progress bar
                    completed_sessions = sum(1 for session in plan['sessions'] if session['completed'])
                    total_sessions = len(plan['sessions'])
                    progress = completed_sessions / total_sessions
                    
                    st.progress(progress)
                    st.write(f"**Progress:** {completed_sessions}/{total_sessions} sessions completed ({progress*100:.1f}%)")
                    
                    # Display sessions
                    st.subheader("Study Sessions")
                    
                    sessions_df = pd.DataFrame(plan['sessions'])
                    sessions_df = sessions_df[['day', 'date', 'time', 'topic', 'duration_minutes', 'completed']]
                    sessions_df.columns = ['Day', 'Date', 'Time', 'Topic', 'Duration (mins)', 'Completed']
                    
                    st.dataframe(sessions_df)
                    
                    # Mark sessions as completed
                    st.subheader("Update Progress")
                    
                    session_day = st.number_input(f"Day", min_value=1, max_value=plan['duration_days'], key=f"day_{i}")
                    session_num = st.number_input(f"Session Number", min_value=1, max_value=3, key=f"session_{i}")
                    
                    if st.button(f"Mark as Completed", key=f"complete_{i}"):
                        # Find the session and mark it as completed
                        for session in plan['sessions']:
                            if session['day'] == session_day and session['session'] == session_num:
                                session['completed'] = True
                                break
                        
                        save_data(notes_data, plans_data, quiz_data)
                        st.success(f"Session marked as completed!")
                        st.rerun()
                    
                    # Delete button
                    if st.button(f"Delete Plan", key=f"delete_plan_{i}"):
                        plans_data.remove(plan)
                        save_data(notes_data, plans_data, quiz_data)
                        st.success(f"Study plan '{plan['name']}' deleted successfully!")
                        st.rerun()
    
    # Create Plan Tab
    with tab2:
        st.subheader("Create New Study Plan")
        
        plan_name = st.text_input("Plan Name")
        plan_subject = st.text_input("Subject")
        plan_topics = st.text_area("Topics (comma-separated)")
        plan_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        plan_duration = st.slider("Duration (days)", min_value=1, max_value=30, value=7)
        
        if st.button("Create Plan"):
            if plan_name and plan_subject and plan_topics:
                # Generate study plan
                sessions = generate_study_plan(plan_subject, plan_topics, plan_difficulty, plan_duration)
                
                # Create a new plan
                new_plan = {
                    "id": str(time.time()),
                    "name": plan_name,
                    "subject": plan_subject,
                    "difficulty": plan_difficulty,
                    "duration_days": plan_duration,
                    "sessions": sessions,
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                plans_data.append(new_plan)
                save_data(notes_data, plans_data, quiz_data)
                
                st.success(f"Study plan '{plan_name}' created successfully!")
                # Clear the form
                st.rerun()
            else:
                st.error("Please provide a name, subject, and topics for the study plan.")

def take_quiz(notes_data, plans_data, quiz_data):
    """Take quizzes based on notes"""
    st.header("Quiz")
    
    # Tabs for quiz actions
    tab1, tab2 = st.tabs(["Take Quiz", "Quiz History"])
    
    # Take Quiz Tab
    with tab1:
        if not notes_data:
            st.info("No notes available. Add notes to take a quiz!")
        else:
            # Filter notes by subject
            subjects = list(set(note.get('subject', 'General') for note in notes_data))
            selected_subject = st.selectbox("Select Subject", ["All"] + subjects)
            
            filtered_notes = notes_data
            if selected_subject != "All":
                filtered_notes = [note for note in notes_data if note.get('subject', 'General') == selected_subject]
            
            # Check if we have enough notes
            if len(filtered_notes) < 1:
                st.warning(f"No notes available for subject: {selected_subject}. Please add notes or select another subject.")
            else:
                # Quiz options - ensure max_value is greater than min_value
                max_questions = max(2, min(10, len(filtered_notes)))  # At least 2
                default_value = min(5, len(filtered_notes))
                
                # Ensure default doesn't exceed max
                default_value = min(default_value, max_questions)
                
                # Use a number_input instead of slider if only 1 question is possible
                if max_questions <= 1:
                    num_questions = 1
                    st.info("Only 1 question possible with current notes.")
                else:
                    num_questions = st.slider("Number of Questions", 
                                          min_value=1, 
                                          max_value=max_questions, 
                                          value=default_value)
                
                # Generate quiz button
                if st.button("Generate Quiz"):
                    # Generate quiz questions
                    questions = generate_quiz(filtered_notes, num_questions)
                    
                    if questions:
                        # Store questions in session state
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_subject = selected_subject
                        
                        st.success(f"Quiz generated with {len(questions)} questions!")
                        st.rerun()
                    else:
                        st.error("Unable to generate quiz questions. Please add more notes.")
            
            # Display quiz questions
            if 'quiz_questions' in st.session_state:
                st.subheader(f"Quiz on {st.session_state.quiz_subject}")
                
                for i, question in enumerate(st.session_state.quiz_questions):
                    st.write(f"**Question {i+1}:** {question['question']}")
                    
                    # Radio buttons for options
                    selected_option = st.radio(
                        f"Select answer for Question {i+1}",
                        question['options'],
                        key=f"question_{i}"
                    )
                    
                    # Store answer
                    st.session_state.quiz_answers[i] = selected_option
                
                if st.button("Submit Quiz"):
                    # Calculate score
                    score = 0
                    results = []
                    
                    for i, question in enumerate(st.session_state.quiz_questions):
                        selected_answer = st.session_state.quiz_answers.get(i)
                        correct_answer = question['correct_answer']
                        
                        is_correct = (selected_answer == correct_answer)
                        if is_correct:
                            score += 1
                        
                        results.append({
                            "question": question['question'],
                            "selected": selected_answer,
                            "correct": correct_answer,
                            "is_correct": is_correct
                        })
                    
                    # Calculate percentage
                    percentage = (score / len(st.session_state.quiz_questions)) * 100
                    
                    # Save quiz results
                    quiz_result = {
                        "id": str(time.time()),
                        "subject": st.session_state.quiz_subject,
                        "questions": len(st.session_state.quiz_questions),
                        "score": percentage,
                        "results": results,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    quiz_data.append(quiz_result)
                    save_data(notes_data, plans_data, quiz_data)
                    
                    # Store results in session state
                    st.session_state.quiz_result = quiz_result
                    
                    # Clear quiz questions
                    del st.session_state.quiz_questions
                    del st.session_state.quiz_answers
                    del st.session_state.quiz_subject
                    
                    st.success(f"Quiz submitted successfully! Your score: {percentage:.1f}%")
                    st.rerun()
            
            # Display quiz results
            if 'quiz_result' in st.session_state:
                result = st.session_state.quiz_result
                
                st.subheader("Quiz Results")
                st.write(f"**Subject:** {result['subject']}")
                st.write(f"**Score:** {result['score']:.1f}%")
                st.write(f"**Date:** {result['created_at']}")
                
                # Display results
                for i, question_result in enumerate(result['results']):
                    with st.expander(f"Question {i+1}"):
                        st.write(f"**Question:** {question_result['question']}")
                        st.write(f"**Your Answer:** {question_result['selected']}")
                        st.write(f"**Correct Answer:** {question_result['correct']}")
                        
                        if question_result['is_correct']:
                            st.success("Correct!")
                        else:
                            st.error("Incorrect")
                
                if st.button("Clear Results"):
                    # Clear quiz results
                    del st.session_state.quiz_result
                    st.rerun()
    
    # Quiz History Tab
    with tab2:
        if not quiz_data:
            st.info("No quiz history available. Take a quiz to see your results!")
        else:
            # Display quiz history
            st.subheader("Quiz History")
            
            # Create a dataframe for quiz history
            quiz_df = pd.DataFrame([
                {
                    "id": quiz['id'],
                    "subject": quiz['subject'],
                    "score": quiz['score'],
                    "questions": quiz['questions'],
                    "date": quiz['created_at']
                }
                for quiz in quiz_data
            ])
            
            quiz_df.columns = ['ID', 'Subject', 'Score (%)', 'Questions', 'Date']
            
            st.dataframe(quiz_df)
            
            # Plot quiz history
            st.subheader("Performance Over Time")
            
            fig = px.line(
                quiz_df,
                x='Date',
                y='Score (%)',
                color='Subject',
                markers=True,
                title='Quiz Scores Over Time'
            )
            
            st.plotly_chart(fig)

def search_notes(notes_data, plans_data, quiz_data):
    """Search notes based on keywords"""
    st.header("Search Notes")
    
    if not notes_data:
        st.info("No notes available. Add notes to search!")
    else:
        search_query = st.text_input("Search Keywords")
        
        if search_query:
            # Find similar notes
            similar_notes = find_similar_notes(search_query, notes_data)
            
            if similar_notes:
                st.subheader("Search Results")
                
                for note, similarity in similar_notes:
                    with st.expander(f"{note['title']} ({note.get('subject', 'General')}) - Relevance: {similarity:.2f}"):
                        st.write(f"**Created:** {note.get('created_at', 'Unknown')}")
                        st.write(note['content'])
            else:
                st.info("No matching notes found. Try different keywords.")
        else:
            st.info("Enter keywords to search for notes.")

def settings(notes_data, plans_data, quiz_data):
    """Application settings and user preferences"""
    st.header("Settings")
    
    # User information
    st.subheader("User Information")
    
    # Load user info if available
    user_info_file = 'data/user_info.json'
    try:
        if os.path.exists(user_info_file) and os.path.getsize(user_info_file) > 0:
            with open(user_info_file, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    user_info = json.loads(file_content)
                else:
                    user_info = {
                        "name": "",
                        "education_level": "Undergraduate",
                        "preferred_subjects": []
                    }
        else:
            user_info = {
                "name": "",
                "education_level": "Undergraduate",
                "preferred_subjects": []
            }
    except (json.JSONDecodeError, FileNotFoundError):
        user_info = {
            "name": "",
            "education_level": "Undergraduate",
            "preferred_subjects": []
        }
    
    # User name
    user_name = st.text_input("Name", user_info.get("name", ""))
    
    # Education level
    education_level = st.selectbox(
        "Education Level",
        ["High School", "Undergraduate", "Graduate", "Professional"],
        index=["High School", "Undergraduate", "Graduate", "Professional"].index(user_info.get("education_level", "Undergraduate"))
    )
    
    # Preferred subjects
    subjects = list(set(note.get('subject', 'General') for note in notes_data))
    preferred_subjects = st.multiselect(
        "Preferred Subjects",
        subjects,
        default=user_info.get("preferred_subjects", [])
    )
    
    # Save user info
    if st.button("Save User Information"):
        user_info = {
            "name": user_name,
            "education_level": education_level,
            "preferred_subjects": preferred_subjects
        }
        
        with open(user_info_file, 'w') as f:
            json.dump(user_info, f)
        
        st.success("User information saved successfully!")
    
    # Rest of the function remains the same...
    
    # Export and import data
    st.subheader("Data Management")
    
    # Export data
    if st.button("Export All Data"):
        export_data = {
            "notes": notes_data,
            "study_plans": plans_data,
            "quiz_history": quiz_data,
            "user_info": user_info
        }
        
        export_file = 'data/studzy_export.json'
        with open(export_file, 'w') as f:
            json.dump(export_data, f)
        
        st.success(f"Data exported successfully to {export_file}!")
    
    # Import data
    st.write("Import Data:")
    import_file = st.file_uploader("Upload JSON file", type=['json'])
    
    if import_file is not None:
        try:
            import_data = json.load(import_file)
            
            if st.button("Import Data"):
                # Check if the file has the expected structure
                if all(key in import_data for key in ["notes", "study_plans", "quiz_history"]):
                    # Update data
                    notes_data.extend(import_data["notes"])
                    plans_data.extend(import_data["study_plans"])
                    quiz_data.extend(import_data["quiz_history"])
                    
                    # Save data
                    save_data(notes_data, plans_data, quiz_data)
                    
                    st.success("Data imported successfully!")
                    st.rerun()
                else:
                    st.error("Invalid data structure in the imported file.")
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
    
    # Reset application
    st.subheader("Reset Application")
    
    if st.button("Reset All Data"):
        # Confirm reset
        st.warning("This will delete all your data and cannot be undone!")
        
        if st.button("Yes, I'm sure"):
            # Clear all data
            notes_data.clear()
            plans_data.clear()
            quiz_data.clear()
            
            # Save empty data
            save_data(notes_data, plans_data, quiz_data)
            
            # Clear user info
            if os.path.exists(user_info_file):
                os.remove(user_info_file)
            
            st.success("All data has been reset successfully!")
            st.rerun()

# --- Run the application ---

if __name__ == "__main__":
    main()