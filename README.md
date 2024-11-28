# KSL-interpreter

# Kenyan Sign Language Interpreter Application

## Overview
The **Sign Language Interpreter Application** is a Flask-based web application that translates sign language gestures into text. It uses machine learning models and computer vision techniques to detect, process, and classify sign language gestures in real time. This project aims to bridge the communication gap between individuals using sign language and those who do not understand it.

---

## Features
1. **Real-Time Sign Language Detection**: 
   - Captures video or image input and translates recognized signs into text.

2. **Customizable Sign Labels**: 
   - Includes letters ('A' to 'Z') and predefined words such as "Hello," "Thanks," "Sorry," etc.

3. **User Authentication**: 
   - Secure login and signup functionality using Flask-Login and Flask-Bcrypt.

4. **Live Interaction**: 
   - Supports live feedback using Flask-SocketIO for an interactive user experience.

5. **Email Notifications**: 
   - Sends email-based notifications using Flask-Mail.

---

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Flask-SocketIO, Flask-SQLAlchemy
- **Machine Learning**: TensorFlow, Mediapipe
- **Database**: SQLite (or configurable to other RDBMS)
- **Libraries**:
  - `opencv-python` for image and video processing.
  - `mediapipe` for hand landmark detection.
  - `flask-bcrypt` for password hashing.
  - `email-validator` for user email validation.

---

## Installation

### Prerequisites
1. Python 3.9+ (recommended version)
2. Pip (Python package manager)
3. A virtual environment for dependency management.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <project_directory>
