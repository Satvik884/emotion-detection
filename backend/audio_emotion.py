import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import gdown 

# Load the pre-trained emotion detection model and label encoder
model_path = 'my_model.h5'
if not os.path.exists(model_path):
    # Google Drive direct download link (replace with your own)
    gdown.download("https://drive.google.com/file/d/1SuaUJpR9Dla7tucNWF1Unzc0lL9yEfC8/view?usp=sharing", model_path, quiet=False)
audio_model = load_model(model_path)
label_encoder = LabelEncoder()
label_encoder.fit(["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"])

def extract_audio_features(audio_path: str):
    """Extract features from audio file at the given file path."""
    y, sr = librosa.load(audio_path, sr=22050)    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features = np.concatenate([mfcc, chroma, spec_contrast], axis=0)
    print(features.shape)
    # Padding or truncating to match model's expected input shape
    max_length = 254
    if features.shape[1] < max_length:
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_length]

    features = np.expand_dims(np.transpose(features), axis=-1)
    features = np.expand_dims(features, axis=0)
    return features

def predict_audio_emotion(audio_path: str):
    audio_features = extract_audio_features(audio_path)
    audio_predictions = audio_model.predict(audio_features)
    audio_pred_label = np.argmax(audio_predictions, axis=-1)
    audio_pred_emotion = label_encoder.inverse_transform(audio_pred_label)[0]
    audio_confidence = np.max(audio_predictions)    
    return audio_pred_emotion,audio_confidence
