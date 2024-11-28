from socket import SocketIO
from wsgiref.simple_server import WSGIServer
from flask import Flask, jsonify, render_template, url_for, redirect, flash, session, request, Response, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from flask_socketio import SocketIO, emit
import socketio
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, Email, EqualTo
from flask_bcrypt import Bcrypt
from datetime import datetime
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer
from flask_mail import Message, Mail
import random
import re
import json
import subprocess
import psutil
import traceback
import logging
import cv2
import time
import signal
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf  
import os
from tensorflow.keras.preprocessing import image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")



CORS(app)  

# -------------------Encrypt Password using Hash Func-------------------
bcrypt = Bcrypt(app)

# -------------------Database Model Setup-------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['UPLOAD_FOLDER'] = './data'
serializer = Serializer(app.config['SECRET_KEY'])
db = SQLAlchemy(app)
app.app_context().push()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -------------------mail configuration----------------------------------
app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_PORT"] = 587
app.config["MAIL_USERNAME"] = 'handssignify@gmail.com'
app.config["MAIL_PASSWORD"] = 'ttbylakctxvvvnxe'
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
mail = Mail(app)
# --------------------------------------------------------


@login_manager.user_loader
def load_user(user_id):
    # Updated code to avoid the deprecation warning
    return db.session.get(User, int(user_id))

# -------------------Database Model-------------------


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), nullable=False, unique=True)
    email = db.Column(db.String(30), nullable=False)
    password = db.Column(db.String(80), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
# ----------------------------------------------------

# -------------------Welcome or Home Page-------------

@app.route('/', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('home.html')
# ----------------------------------------------------

# -------------------feed back Page-----------------------
@app.route('/feed', methods=['GET', 'POST'])
@login_required
def feed():
    return render_template('feed.html')
# ----------------------------------------------------


# -------------------Discover More Page---------------
@app.route('/discover_more', methods=['GET', 'POST']) 
def discover_more():
    return render_template('discover_more.html')
# ----------------------------------------------------

# -------------------Guide Page-----------------------
@app.route('/guide', methods=['GET', 'POST'])
def guide():
    return render_template('guide.html')
# ----------------------------------------------------


# -------------------Login Page-------------------
class LoginForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    password = PasswordField(label='password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    # Check if the user has registered before showing the login form
    if 'registered' in session and session['registered']:
        session.pop('registered', None)
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data) and User.query.filter_by(email=form.email.data).first():
            login_user(user)
            flash('Login successfully.', category='success')
            name = form.username.data
            session['name'] = name
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash(f'Login unsuccessful for {form.username.data}.', category='danger')
    return render_template('login.html', form=form)
# ----------------------------------------------------


# -------------------Dashboard or Logged Page-------------------
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if 'logged_in' in session and session['logged_in']:
        name = session.get('name')
        # character = session.get('character')
        # templs = ['detect_characters.html', 'dashboard.html']
        return render_template('dashboard.html', name=name)
    return redirect(url_for('login'))
# ----------------------------------------------------

# -------------------About Page-----------------------
@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')
# ----------------------------------------------------

# -------------------Logged Out Page-------------------

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    session.clear()
    logout_user()
    flash('Account Logged out successfully.', category='success')
    return redirect(url_for('login'))
# ----------------------------------------------------

# -------------------Register Page-------------------

class RegisterForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    password = PasswordField(label='password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    confirm_password = PasswordField(label='confirm_password', validators=[InputRequired(), EqualTo('password')], render_kw={"placeholder": "Confirm Password"})
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            flash('That Username already exists. Please choose a different one.', 'danger')
            raise ValidationError('That username already exists. Please choose a different one.')


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data,email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        # Set a session variable to indicate successful registration
        session['registered'] = True
        flash(f'Account Created for {form.username.data} successfully.', category='success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)
# ----------------------------------------------------

# -------------------Update or reset Email Page-------------------


class ResetMailForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Old Email"})
    new_email = StringField(label='new_email', validators=[InputRequired(), Email()], render_kw={"placeholder": "New Email"})
    password = PasswordField(label='password', validators=[InputRequired()], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login', validators=[InputRequired()])


@app.route('/reset_email', methods=['GET', 'POST'])
@login_required
def reset_email():
    form = ResetMailForm()
    if 'logged_in' in session and session['logged_in']:
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user and bcrypt.check_password_hash(user.password, form.password.data) and User.query.filter_by(email=form.email.data).first():
                user.email = form.new_email.data  # Replace old email with new email
                db.session.commit()
                flash('Email reset successfully.', category='success')
                session.clear()
                return redirect(url_for('login'))
            else:
                flash('Invalid email, password, or combination.', category='danger')

        return render_template('reset_email.html', form=form)
    return redirect(url_for('login'))
# --------------------------------------------------------------

# -------------------Forgot Password With OTP-------------------

class ResetPasswordForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    submit = SubmitField('Submit', validators=[InputRequired()])


class ForgotPasswordForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    new_password = PasswordField(label='new_password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "New Password"})
    confirm_password = PasswordField(label='confirm_password', validators=[InputRequired(), EqualTo('new_password')], render_kw={"placeholder": "Confirm Password"})
    otp = StringField(label='otp', validators=[InputRequired(), Length(min=6, max=6)], render_kw={"placeholder": "Enter OTP"})
    submit = SubmitField('Submit', validators=[InputRequired()])


@staticmethod
def send_mail(name, email, otp):
    msg = Message('Reset Email OTP Password',sender='handssignify@gmail.com', recipients=[email])
    msg.body = "Hii " + name + "," + "\nYour email OTP is :"+str(otp)
    mail.send(msg)


    # Generate your OTP logic here
def generate_otp():
    return random.randint(100000, 999999)


@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    otp = generate_otp()
    session['otp'] = otp
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and User.query.filter_by(email=form.email.data).first():
            send_mail(form.username.data, form.email.data, otp)
            flash('Reset Request Sent. Check your mail.', 'success')
            return redirect(url_for('forgot_password'))
        else:
            flash('Email and username combination is not exist.', 'danger')
    return render_template('reset_password_request.html', form=form)


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        otp = request.form['otp']
        valid = (otp == request.form['otp'])

        if valid:
            user = User.query.filter_by(username=form.username.data).first()
            if user and User.query.filter_by(email=form.email.data).first():
                user.password = bcrypt.generate_password_hash(form.new_password.data).decode('utf-8')
                db.session.commit()
                flash('Password Changed Successfully.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Email and username combination is not exist.', 'danger')
        else:
            flash("OTP verification failed.", 'danger')
    return render_template('forgot_password.html', form=form)
# ---------------------------------------------------------------

# ------------------------- Update Password ---------------------

class UpdatePasswordForm(FlaskForm):
    username = StringField(label='username', validators=[InputRequired()], render_kw={"placeholder": "Username"})
    email = StringField(label='email', validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    new_password = PasswordField(label='new_password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "New Password"})
    confirm_password = PasswordField(label='confirm_password', validators=[InputRequired(), EqualTo('new_password')], render_kw={"placeholder": "Confirm Password"})
    submit = SubmitField('Submit', validators=[InputRequired()])


@app.route('/update_password', methods=['GET', 'POST'])
@login_required
def update_password():
    form = UpdatePasswordForm()
    if form.validate_on_submit() and 'logged_in' in session and session['logged_in']:

            user = User.query.filter_by(username=form.username.data).first()
            if user and User.query.filter_by(email=form.email.data).first():
                user.password = bcrypt.generate_password_hash(form.new_password.data).decode('utf-8')
                db.session.commit()
                flash('Password Changed Successfully.', 'success')
                session.clear()
                return redirect(url_for('login'))
            else:
                flash("Username and email combination is not exist.", 'danger')
    return render_template('update_password.html', form=form)
# -----------------------------  end  ---------------------------


# --------------------------- Machine Learning ------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'your_secret_key'

# Function to check if the file is an allowed image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to preprocess the image and make a prediction
def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))  # Assuming the model was trained with 64x64 images
    img_array = image.img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print("Expected input shape:", model.input_shape)
    print("Actual image shape:", img_array.shape)

    return img_array

model = load_model('model.h5')

# Route for handling sign prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file temporarily
        file_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(file_path)

        # Preprocess the image and make a prediction
        img_array = prepare_image(file_path)
        prediction = model.predict(img_array)
        prediction = model.predict(img_array)
        print("Raw predictions:", prediction)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print("Predicted class:", predicted_class)

        # Assuming labels are the same as in your original code
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 
                  'yes', 'no', 'please', 'sorry']
        
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = labels[predicted_class]

        # Return the prediction as JSON response
        return jsonify({'prediction': predicted_label})
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction. Please try again later.'}), 500

        # If the predicted sign is not recognized, notify the user
        if predicted_label not in labels:
            flash('This sign is not recognized in the trained model. Please try again with another image.')
            return redirect(url_for('dashboard'))

        # Show the prediction to the user
        flash(f"Predicted sign: {predicted_label}")
        return redirect(url_for('dashboard'))
    
    else:
        flash('Invalid file format. Please upload a .png, .jpg, or .jpeg file.')
        return redirect(url_for('dashboard'))

# The route for the dashboard page
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard_view():
    return render_template('dashboard.html')

@app.route('/data/<path:filename>')
def serve_file(filename):
    return send_from_directory('./data', filename)

@app.route('/data/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

sign_data = {
    "A": {"image": "A.jpg", "description": "Make a fist with your thumb resting on the side of your fingers."},
    "B": {"image": "B.jpg", "description": "Extend your fingers straight upward and tuck your thumb into your palm."},
    "C": {"image": "C.jpg", "description": "Curve your fingers into the shape of the letter 'C'."},
    "D": {"image": "D.jpg", "description": "Touch your thumb to your middle, ring, and pinky fingers while extending your index finger upward."},
    "E": {"image": "E.jpg", "description": "Curve all your fingers tightly into your palm, with the thumb resting in front."},
    "F": {"image": "F.jpg", "description": "Form a circle with your thumb and index finger while extending the other fingers upward."},
    "G": {"image": "G.jpg", "description": "Point your thumb and index finger outward, keeping the other fingers curled into the palm."},
    "H": {"image": "H.jpg", "description": "Extend your index and middle fingers horizontally, with the other fingers curled into your palm."},
    "I": {"image": "I.jpg", "description": "Extend your pinky finger while curling the other fingers into your palm."},
    "J": {"image": "J.jpg", "description": "Extend your pinky finger and trace the shape of a 'J' in the air."},
    "K": {"image": "K.jpg", "description": "Extend your index and middle fingers upward, with your thumb placed between them."},
    "L": {"image": "L.jpg", "description": "Form an 'L' shape with your thumb and index finger while curling the other fingers into your palm."},
    "M": {"image": "M.jpg", "description": "Place your thumb under your index, middle, and ring fingers while extending the pinky."},
    "N": {"image": "N.jpg", "description": "Place your thumb under your index and middle fingers while extending the other fingers."},
    "O": {"image": "O.jpg", "description": "Curl your fingers and thumb into a circle to form the shape of an 'O'."},
    "P": {"image": "P.jpg", "description": "Form a 'K' shape and tilt it downward."},
    "Q": {"image": "Q.jpg", "description": "Point your index finger and thumb downward while curling the other fingers into your palm."},
    "R": {"image": "R.jpg", "description": "Cross your index and middle fingers, curling the others into your palm."},
    "S": {"image": "S.jpg", "description": "Make a fist with your thumb tucked in front of your fingers."},
    "T": {"image": "T.jpg", "description": "Place your thumb between your index and middle fingers while curling the others into your palm."},
    "U": {"image": "U.jpg", "description": "Extend your index and middle fingers upward together, with the other fingers curled."},
    "V": {"image": "V.jpg", "description": "Extend your index and middle fingers upward, forming a 'V' shape."},
    "W": {"image": "W.jpg", "description": "Extend your index, middle, and ring fingers upward, forming a 'W' shape."},
    "X": {"image": "X.jpg", "description": "Curl your index finger into a hook shape, with the thumb extended."},
    "Y": {"image": "Y.jpg", "description": "Extend your thumb and pinky finger outward, curling the other fingers into your palm."},
    "Z": {"image": "Z.jpg", "description": "Draw the shape of a 'Z' in the air using your index finger."},
    "HELLO": {"image": "hello.jpg", "description": "Raise your hand and salute."},
    "THANKS": {"image": "thank_you.jpg", "description": "Place your fingers on your chin and move them outward."},
    "SORRY": {"image": "sorry.jpg", "description": "Make a fist and rub it in a circular motion over your chest."},
    "PLEASE": {"image": "please.jpg", "description": "Rub your open palm in a circular motion on your chest."},
    "YES": {"image": "yes.jpg", "description": "Make a fist and nod it up and down, as if nodding your head."},
    "NO": {"image": "no.jpg", "description": "Extend your index and middle fingers together, then tap them against your thumb."}
    }


def get_sign_info(sign_name): 
    # Synonym mapping to handle different variants of the same sign (case-insensitive)
    synonym_map = {
        "hello": ["hello", "hi", "hey", "greetings", "howdy"],
        "thanks": ["thanks", "thank you", "many thanks", "much appreciated"],
        "sorry": ["sorry", "apologies", "my apologies", "I’m sorry", "pardon me", "excuse me", "forgive me", "I regret", "I beg your pardon"],
        "please": ["please", "kindly", "if you please", "would you mind", "could you", "would you be so kind", "I'd appreciate it"],
        "yes": ["yes", "yeah", "yep", "affirmative", "certainly", "of course", "sure", "indeed"],
        "no": ["no", "nope", "nah", "negative", "not at all", "no way", "absolutely not", "never"]
    }

    # Normalize the input sign name (case-insensitive)
    sign_name = sign_name.lower()

    # Loop through synonyms to check if the input matches any, and normalize to the canonical form
    for canonical_sign, synonyms in synonym_map.items():
        if sign_name in [synonym.lower() for synonym in synonyms]:
            sign_name = canonical_sign  # Use the canonical sign name
            break

    # Now, use the normalized sign_name (converted to uppercase) to lookup in sign_data
    sign_name = sign_name.upper()

    # Dynamic folder lookup
    data_folder = './data'
    sign_folder_path = os.path.join(data_folder, sign_name)

    # Check if folder exists for the given sign
    if os.path.exists(sign_folder_path) and os.path.isdir(sign_folder_path):
        # Get the first image in the folder
        image_files = [f for f in os.listdir(sign_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            # Correct image URL generation
            image_url = url_for('serve_file', filename=f'{sign_name}/{image_files[0]}')  # Correct path format
        else:
            image_url = url_for('serve_file', filename='not_found.jpg')
    else:
        image_url = url_for('serve_file', filename='not_found.jpg')

    # Use predefined description if available in sign_data, or a generic one
    description = sign_data.get(sign_name, {}).get("description")

    if not description:
        description = f"This is the sign for '{sign_name}'."  # Fallback message if no description exists

    return {"image": image_url, "description": description}

# -----------------------------  how to sign  -------------------------------------------------------------------------

# Route for the dashboard
@app.route('/how_to_sign_page')
def how_to_sign_page():
    #print("Navigated to How to Sign page")
    return render_template('how_to_sign.html')

@app.route('/how_to_sign', methods=['GET'])

def how_to_sign():
    sign_name = request.args.get('sign')
    
    if not sign_name:
        return jsonify({"success": False, "message": "Sign name is required."})
    
    # Assume sign_data is a dictionary that holds the sign info
    sign_info = get_sign_info(sign_name)
    
    if sign_info:
        return jsonify({
            "success": True,
           'sign': {
                'name': sign_name,
                'image_url': sign_info["image"],
                'description': sign_info['description']
            }
        })
    else:
        return jsonify({"success": False, "message": "Sign not found."})

# -----------------------------  webcam  -----------------------------------------------------------

prediction_process = None

logging.basicConfig(level=logging.DEBUG)

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/start_realtime', methods=['GET'])
def start_realtime():
    global prediction_process 
    try:
        if prediction_process is None or prediction_process.poll() is not None:
            # Start the real-time prediction process
            prediction_process = subprocess.Popen(['python', 'stand_alone.py'])
            return jsonify(status='Real-time prediction started')
        else:
            return jsonify(status='Real-time prediction already running')
    except FileNotFoundError:
        return jsonify(status="Error: stand_alone.py file not found"), 500
    except subprocess.SubprocessError as e:
        return jsonify(status=f"Error in subprocess: {str(e)}"), 500
    except Exception as e:
        return jsonify(status=f"Unexpected error: {str(e)}"), 500

@socketio.on('prediction_data')
def handle_prediction_data(data):
    print(f"Broadcasting prediction: {data}")
    emit('prediction_data', data, broadcast=True)

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('server_response', {'data': 'Connection established'})

@socketio.on('disconnect')
def handle_disconnect():
    print("WebSocket connection closed")


@app.route('/stop_realtime', methods=['GET'])
def stop_realtime():
    global prediction_process 
    try:
        if prediction_process is not None:
            logging.debug("Terminating the real-time prediction process...")
            prediction_process.kill()  # Forcefully stop the process
            prediction_process = None
            return jsonify(status='Real-time prediction stopped')
        else:
            return jsonify(status='No real-time prediction is running')
    except Exception as e:
        logging.error(f"Error stopping process: {str(e)}")
        return jsonify(status=f"Error stopping process: {str(e)}"), 500

# -----------------------------  end  ---------------------------
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)  
