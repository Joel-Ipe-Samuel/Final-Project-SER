import torch.nn as nn
import numpy as np
import librosa
import numpy as np

# Enhanced Neural Network Model with Conv1D, LSTM, and Attention
class EnhancedSERModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedSERModel, self).__init__()
        # Convolutional layers for extracting features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # LSTM layers for capturing temporal information
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True)
        
        # Attention mechanism to focus on relevant parts of the sequence
        self.attention = nn.MultiheadAttention(embed_dim=128*2, num_heads=8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128*2, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout layers
        self.dropout_conv = nn.Dropout(0.5)  # Dropout after convolution
        self.dropout_lstm = nn.Dropout(0.5)  # Dropout after LSTM
        self.dropout_fc = nn.Dropout(0.5)    # Dropout after first fully connected layer

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # Apply Conv1D
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)  # Apply dropout after convolution
        
        # Apply LSTM
        x = x.permute(0, 2, 1)  # Rearrange to (batch, seq_len, feature)
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)  # Apply dropout after LSTM

        # Apply Attention
        x = x.permute(1, 0, 2)  # Rearrange for attention (seq_len, batch, feature)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, feature)

        # Global Average Pooling
        x = x.mean(dim=1)  # (batch, feature)

        # Fully connected layers
        x = self.leaky_relu(self.bn2(self.fc1(x)))
        x = self.dropout_fc(x)  # Apply dropout after fully connected layer
        x = self.fc2(x)
        return x
    
# Function to extract Log-Mel Spectrogram
def extract_log_mel_spectrogram(signal, sr, n_mels=40, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

# Function to extract MFCCs
def extract_mfcc(signal, sr, n_mfcc=128):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc, n_mfcc  # Return MFCCs and their count

# Function to extract Chroma features
def extract_chroma(signal, sr, n_fft=2048, hop_length=512):
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return chroma, chroma.shape[0]  # Return Chroma features and their count

# Function to extract Spectral Contrast
def extract_spectral_contrast(signal, sr, n_fft=2048, hop_length=512):
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return spectral_contrast, spectral_contrast.shape[0]  # Return Spectral Contrast features and their count

# Function to extract audio features
def extract_features(signal, sr):
    log_mel_spectrogram = extract_log_mel_spectrogram(signal, sr)
    mfcc, mfcc_count = extract_mfcc(signal, sr)
    chroma, chroma_count = extract_chroma(signal, sr)
    spectral_contrast, spectral_contrast_count = extract_spectral_contrast(signal, sr)

    combined_features = np.hstack([
        np.mean(log_mel_spectrogram.T, axis=0),
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0)
    ])
       # Calculate total feature count dynamically
    total_feature_count = (
        log_mel_spectrogram.shape[0] + 
        mfcc_count + 
        chroma_count + 
        spectral_contrast_count
    )
    
    return combined_features, total_feature_count