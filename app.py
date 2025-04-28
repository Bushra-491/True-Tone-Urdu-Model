import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import logging
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Enable Flask logging
logging.basicConfig(level=logging.DEBUG)
app.logger.info("Flask app started")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best_urdu_deep_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def convert_to_wav(in_path: str, out_path: str) -> str:
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

def remove_noise(y: np.ndarray, sr: int) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

def augment_audio(y: np.ndarray, sr: int) -> list:
    aug = []
    aug.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2))
    aug.append(librosa.effects.time_stretch(y=y, rate=1.2))
    aug.append(y + 0.005 * np.random.randn(len(y)))
    aug.append(librosa.effects.time_stretch(y=y, rate=0.8))
    aug.append(y * np.random.uniform(0.7, 1.3))
    aug.append(np.convolve(y, np.ones(200)/200, mode='same'))
    b, a = butter(6, [300/(sr/2), 3400/(sr/2)], btype='band')
    aug.append(lfilter(b, a, y))
    return aug

# Feature extraction
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rmse])

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Save uploaded audio
        temp_input_path = "temp_input"
        os.makedirs(temp_input_path, exist_ok=True)
        raw_audio_path = os.path.join(temp_input_path, audio_file.filename)
        audio_file.save(raw_audio_path)

        # Convert to wav
        wav_path = os.path.join(temp_input_path, "converted.wav")
        convert_to_wav(raw_audio_path, wav_path)
        
        # Load and preprocess audio
        y, sr = librosa.load(wav_path, sr=16000)
        y = remove_noise(y, sr)

        # Create augmented versions
        augmented_audios = augment_audio(y, sr)
        all_features = []

        # Extract features from original + augmented audios
        for aug_y in [y] + augmented_audios:
            features = extract_features(aug_y, sr)
            all_features.append(features)

        # Average features
        avg_features = np.mean(all_features, axis=0)

        # Scale features
        scaled_features = scaler.transform([avg_features]).astype(np.float32)

        # Predict
        interpreter.set_tensor(input_details[0]['index'], scaled_features)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Decode prediction
        predicted_label = label_encoder.inverse_transform([np.argmax(output_data)])[0]

        return jsonify({'result': predicted_label})

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            for f in os.listdir(temp_input_path):
                os.remove(os.path.join(temp_input_path, f))
            os.rmdir(temp_input_path)

if __name__ == '__main__':
    app.run(debug=True)
