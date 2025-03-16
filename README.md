# Studzy - AI Study Assistant

## Overview
Studzy is a comprehensive AI-powered study assistant designed to help students organize their learning materials, create effective study plans, and test their knowledge. Built with Python and Streamlit, this application offers an intuitive interface for managing your educational journey.

## Features

### 📝 Note Management
- Create, edit, and delete study notes
- Organize notes by subject
- Search notes using advanced NLP techniques
- View note history and analytics

### 📅 Study Planning
- Generate customized study plans based on your schedule and preferences
- Track progress with interactive progress bars
- Adjust difficulty levels (Easy, Medium, Hard)
- Manage study sessions with session-specific tracking

### 📊 Knowledge Assessment
- Generate quizzes based on your study notes
- Track quiz performance over time
- Analyze strengths and weaknesses
- Review quiz history with detailed analytics

### 🔍 Smart Search
- Find relevant notes using natural language processing
- Search across subjects and topics
- Discover connections between different study materials

## Dashboard
The dashboard provides at-a-glance information about:
- Total number of notes
- Active study plans
- Average quiz scores
- Recent activity
- Upcoming study sessions
- Progress charts

## Technology Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly
- **NLP**: NLTK, Scikit-learn
- **Data Storage**: JSON files

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/studzy.git
cd studzy
```

2. Create a virtual environment (recommended)
```bash
python -m venv studzy-env
source studzy-env/bin/activate  # On Windows: studzy-env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

### Requirements
Create a `requirements.txt` file with the following content:
```
streamlit
pandas
numpy
matplotlib
plotly
scikit-learn
nltk
python-dateutil
```

## Usage
1. Launch the app using `streamlit run app.py`
2. Navigate through the sidebar to access different features
3. Start by creating notes on your study topics
4. Generate study plans based on your schedule
5. Test your knowledge with auto-generated quizzes
6. Track your progress using the dashboard

## Data Management
- All data is stored locally in JSON files
- Export/import functionality for data backup and transfer
- User settings are saved between sessions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- Built with Streamlit, a powerful framework for data applications
- Utilizes NLTK for natural language processing
- Implements advanced quiz generation algorithms

---

