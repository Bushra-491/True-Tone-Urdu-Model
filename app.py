import os
import tempfile
import numpy as np
import librosa
import joblib
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import gradio as gr
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best_urdu_deep_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Recreate exact feature names used during training to prevent StandardScaler misalignment
FEATURE_NAMES = (
    ['MFCC_{}'.format(i+1) for i in range(13)] + \
    ['Chroma_{}'.format(i+1) for i in range(12)] + \
    ['Spectral_Contrast_{}'.format(i+1) for i in range(7)] + \
    ['ZCR'] + ['RMSE']
)

def convert_to_wav(in_path: str, out_path: str) -> str:
    # Let pydub figure out the format automatically to support all file types
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

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rmse])

def predict(audio_path: str) -> str:
    if audio_path is None:
        return "Please upload an audio file."
    
    # Use named temporary file to handle concurrent users safely
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        # Convert any audio standard/extension securely
        convert_to_wav(audio_path, wav_path)
        
        # Load and process
        y, sr = librosa.load(wav_path, sr=16000)
        y = remove_noise(y, sr)
        
        # Features mapping
        augmented_audios = augment_audio(y, sr)
        all_features = [extract_features(aug_y, sr) for aug_y in [y] + augmented_audios]
        avg_features = np.mean(all_features, axis=0)
        
        # FIX: Format as a pandas DataFrame matching Kaggle training structure precisely
        feats_df = pd.DataFrame(avg_features.reshape(1, -1), columns=FEATURE_NAMES)
        scaled_features = scaler.transform(feats_df).astype(np.float32)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], scaled_features)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Translate predictions back to string labels
        predicted_label = label_encoder.inverse_transform([np.argmax(output_data)])[0]
        return f"Prediction: {predicted_label}"
        
    except Exception as e:
        return f"Error: {str(e)}"
        
    finally:
        # Clean up the temporary file immediately
        if os.path.exists(wav_path):
            os.remove(wav_path)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Urdu Audio File"),
    outputs=gr.Text(label="Result"),
    title="TrueTone — Urdu Audio Forgery Detection",
    description="Upload an Urdu audio file to detect whether it is Original, AI Generated, or Combined. Built using a TFLite deep learning model trained on 1,530+ Urdu audio samples with 7-type data augmentation.",
    examples=[],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
