import numpy as np
import scipy.signal
import scipy.fftpack as fft
import socket
import joblib
from collections import deque
from scipy.stats import skew, kurtosis
import pywt
import traceback
import time
import winsound
import keyboard
from sklearn.ensemble import RandomForestClassifier

# ====== Feature Extraction Functions (MATCHING TRAINING EXACTLY) ======
def hjorth_parameters(data):
    """Modified to match training implementation exactly"""
    data = data.astype(np.float64)  # μV conversion added
    first_derivative = np.diff(data, axis=1)
    second_derivative = np.diff(first_derivative, axis=1)
    var_zero = np.var(data, axis=1)
    var_d1 = np.var(first_derivative, axis=1)
    var_d2 = np.var(second_derivative, axis=1)
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return np.vstack((var_zero/1e4, mobility, complexity)).T  # Additional scaling matches training

def shannon_entropy(signal):
    """Unchanged"""
    pdf = np.histogram(signal, bins=10, density=True)[0]
    pdf = pdf[pdf > 0]
    return -np.sum(pdf * np.log2(pdf))

def spectral_entropy(signal_data, fs):
    """Fixed version that avoids naming conflict"""
    f, Pxx = scipy.signal.welch(signal_data, fs=fs, nperseg=min(fs//4, len(signal_data)))
    Pxx_norm = Pxx / (np.sum(Pxx) + 1e-12)
    return -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))

def higuchi_fd(signal, kmax=10):
    """More robust implementation with error handling"""
    N = len(signal)
    L = np.zeros(kmax)
    
    for k in range(1, kmax+1):
        Lk = np.zeros(k)
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:  # Handle case where we don't have enough points
                Lk[m] = 0
                continue
                
            diff = np.abs(np.diff(signal[idx]))
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = len(diff) * k
                Lk[m] = np.sum(diff) * (N-1) / denominator if denominator > 0 else 0
                
        L[k-1] = np.mean(Lk[Lk > 0]) if np.any(Lk > 0) else 0
        
    # Handle case where all L values are zero
    if np.all(L == 0):
        return 0.0
        
    coeffs = np.polyfit(np.log(np.arange(1,kmax+1)), np.log(L + 1e-10), 1)
    return abs(coeffs[0])

def wavelet_features(signal):
    """Matches training exactly"""
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    return np.hstack([np.mean(abs(c)) for c in coeffs])

def phase_locking_value(signal1, signal2):
    """Unchanged"""
    phase_diff = np.angle(scipy.signal.hilbert(signal1)) - np.angle(scipy.signal.hilbert(signal2))
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def zero_crossing_rate(signal):
    """Modified to match training's 2D handling"""
    return np.array([np.sum(np.abs(np.diff(np.sign(ch)))) / (2 * len(ch)) 
                    for ch in signal])[:, np.newaxis]

def root_mean_square(signal):
    """Modified to match training's 2D handling"""
    return np.sqrt(np.mean(signal**2, axis=1))

def peak_frequency(signal, fs):
    """Calculate peak frequency for each channel in a 2D array"""
    peak_freqs = []
    for ch in signal:
        f, Pxx = scipy.signal.welch(ch, fs=fs, nperseg=min(fs//4, len(ch)))
        peak_freqs.append(f[np.argmax(Pxx)])
    return np.array(peak_freqs)[:, np.newaxis]

def spectral_edge_frequency(signal, fs, edge=0.95):
    """Modified to match training's 2D handling"""
    sefs = []
    for ch in signal:
        f, Pxx = scipy.signal.welch(ch, fs=fs, nperseg=min(fs//4, len(ch)))
        cumsum = np.cumsum(Pxx)
        sefs.append(f[np.where(cumsum >= edge * cumsum[-1])[0][0]])
    return np.array(sefs)[:, np.newaxis]

def spectral_skewness(signal, fs):
    """Modified to match training's 2D handling"""
    skews = []
    for ch in signal:
        f, Pxx = scipy.signal.welch(ch, fs=fs, nperseg=min(fs//4, len(ch)))
        skews.append(skew(Pxx))
    return np.array(skews)[:, np.newaxis]

def spectral_kurtosis(signal, fs):
    """Modified to match training's 2D handling"""
    kurtoses = []
    for ch in signal:
        f, Pxx = scipy.signal.welch(ch, fs=fs, nperseg=min(fs//4, len(ch)))
        kurtoses.append(kurtosis(Pxx))
    return np.array(kurtoses)[:, np.newaxis]

def cross_correlation(signal1, signal2):
    """Unchanged"""
    return np.corrcoef(signal1, signal2)[0, 1]

def fractal_dimension(signal):
    """Matches training"""
    return higuchi_fd(signal, kmax=10)

# ====== Main Processing Class ======
class EEGMovementClassifier:
    def __init__(self, model_path):
        """Load the saved model package"""
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.selected_indices = self.model_package['selected_indices']
        self.optimal_thresholds = self.model_package['optimal_thresholds']
        self.class_labels = self.model_package['class_labels']
        self.feature_labels = self.model_package['feature_labels']
        
        # Reconstruct filter
        self.b_filter = np.array(self.model_package['filters']['b'])
        self.a_filter = np.array(self.model_package['filters']['a'])
        
        # Get expected parameters
        self.fs = 500
        self.num_channels = 8
        self.window_size = 1.0  # seconds
        self.samples_per_window = int(self.fs * self.window_size)
        
        print(f"✅ Model loaded with {len(self.class_labels)} classes")
        print(f"Using {len(self.selected_indices)} selected features")

    def preprocess(self, raw_eeg_data):
        """Match training preprocessing exactly"""
        processed_data = raw_eeg_data.astype(np.float64)
        processed_data = np.clip(processed_data, -22.5, 22.5)
        
        # Whitening transform
        if hasattr(self, 'whitening_matrix'):
            flattened = processed_data.reshape(self.num_channels, -1)
            whitened = np.dot(self.whitening_matrix, flattened)
            processed_data = whitened.reshape(processed_data.shape)
        
        filtered = scipy.signal.filtfilt(self.b_filter, self.a_filter, processed_data)
        return filtered

    def _calc_bandpower(self, data, band):
        """Matches EXACTLY how bandpower was calculated during training"""
        # 1. Same μV conversion and precision
        data = data.astype(np.float64)
        
        # 2. Same Welch parameters as training
        f, Pxx = scipy.signal.welch(data, 
                            fs=self.fs, 
                            nperseg=min(self.fs//4, data.shape[1]),  # Exactly as in training
                            axis=1)  # Process all channels at once
        
        # 3. Same bandpower calculation logic
        mask = (f >= band[0]) & (f <= band[1])
        band_power = np.sum(Pxx[:, mask], axis=1) / (np.sum(Pxx, axis=1) + 1e-12)
        
        return band_power.reshape(-1, 1)  # Return as column vector to match feature extraction

    def extract_features(self, window_data):
        """EXACTLY matches training feature extraction pipeline"""
        # 1. Preprocessing (must match training)
        processed_data = window_data.astype(np.float64) # Same conversion
        processed_data = np.clip(processed_data, -22.5, 22.5)  # Same clipping
        
        # 2. Filtering (using the saved filter coefficients)
        filtered = scipy.signal.filtfilt(self.b_filter, self.a_filter, processed_data)
        
        # Initialize feature list in EXACT SAME ORDER as training
        features = []
        
        # 3. Time-domain features (identical to training)
        features.append(np.mean(filtered, axis=1, keepdims=True))
        features.append(np.log1p(np.var(filtered, axis=1, keepdims=True)))
        features.append(skew(filtered, axis=1, keepdims=True))
        features.append(kurtosis(filtered, axis=1, keepdims=True))
        features.append(np.median(filtered, axis=1, keepdims=True))
        features.append(np.max(filtered, axis=1, keepdims=True))
        features.append(np.min(filtered, axis=1, keepdims=True))
        features.append(np.log1p(np.sum((filtered)**2, axis=1, keepdims=True)))
        
        # 4. Frequency-domain features (careful with welch!)
        fft_mag = np.abs(fft.rfft(filtered, axis=1))
        features.append(np.log1p(np.mean(fft_mag, axis=1, keepdims=True)))
        features.append(np.log1p(np.var(fft_mag, axis=1, keepdims=True)))
        
        # 5. Bandpower features (critical - must match training exactly)
        features.append(self._calc_bandpower(filtered, [5, 8]).reshape(-1, 1))
        features.append(self._calc_bandpower(filtered, [8, 13]).reshape(-1, 1)) 
        features.append(self._calc_bandpower(filtered, [13, 30]).reshape(-1, 1))
        features.append(self._calc_bandpower(filtered, [30, 60]).reshape(-1, 1))
        
        # 6. Non-linear features
        features.append(hjorth_parameters(filtered))
        features.append(np.apply_along_axis(shannon_entropy, 1, filtered)[:, np.newaxis])
        features.append(np.apply_along_axis(lambda x: spectral_entropy(x, self.fs), 1, filtered)[:, np.newaxis])
        features.append(np.apply_along_axis(fractal_dimension, 1, filtered)[:, np.newaxis])
        features.append(np.vstack([wavelet_features(ch) for ch in filtered]))
        
        # 7. Other features
        features.append(zero_crossing_rate(filtered))
        features.append(root_mean_square(filtered)[:, np.newaxis])
        features.append(peak_frequency(filtered, self.fs))
        features.append(spectral_edge_frequency(filtered, self.fs))
        features.append(spectral_skewness(filtered, self.fs))
        features.append(spectral_kurtosis(filtered, self.fs))
        
        # 8. Connectivity features - MUST MATCH TRAINING
        plv_features = []
        cross_corr_features = []
        
        for i in range(self.num_channels):
            for j in range(i+1, self.num_channels):
                plv_features.append(phase_locking_value(filtered[i], filtered[j]))
                cross_corr_features.append(cross_correlation(filtered[i], filtered[j]))
        
        # Tile connectivity features to match training (8x replication)
        plv_tiled = np.tile(plv_features, (self.num_channels, 1))  # Shape (8, 28)
        cross_corr_tiled = np.tile(cross_corr_features, (self.num_channels, 1))  # Shape (8, 28)
        
        features.append(plv_tiled)
        features.append(cross_corr_tiled)
        
        # Combine all features and verify count
        feature_vector = np.concatenate([f for f in features], axis=1).flatten()
        
        if len(feature_vector) != 688:  # Must match training dimension
            raise ValueError(f"Feature dimension mismatch! Expected 688, got {len(feature_vector)}")
        
        return feature_vector


    def predict(self, raw_eeg_window):
        """Make prediction with dynamic thresholding and confidence checking"""
        # Validate input
        if raw_eeg_window.shape != (self.num_channels, self.samples_per_window):
            raise ValueError(f"Expected shape {(self.num_channels, self.samples_per_window)}, got {raw_eeg_window.shape}")
        
        # Preprocess
        filtered = self.preprocess(raw_eeg_window)
        
        # Extract features
        features = self.extract_features(filtered)
        
        # Scale and select features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_selected = features_scaled[:, self.selected_indices]
        
        # Get probabilities
        proba = self.model.predict_proba(features_selected)[0]
        
        # Confidence check - reject if max probability isn't sufficiently high
        max_prob = np.max(proba)
        if max_prob < 0.15:  # Adjust this threshold based on your needs
            return "StopThoughts"
        
        # Apply class-specific thresholds
        thresholded_proba = np.zeros_like(proba)
        for i, class_label in enumerate(self.class_labels):
            threshold = self.optimal_thresholds[class_label]
            thresholded_proba[i] = max(0, proba[i] - threshold)
        
        # If no class meets threshold, return StopThoughts
        if np.sum(thresholded_proba) == 0:
            return "StopThoughts"
        
        # Normalize and return highest class
        thresholded_proba /= np.sum(thresholded_proba)
        return self.class_labels[np.argmax(thresholded_proba)]

CLASS_KEYS = {
    'q': 'OpenThoughts',
    'w': 'CloseThoughts',
    'e': 'TwistLeftThoughts',
    'r': 'TwistRightThoughts',
    't': 'SmallUpThoughts',
    'y': 'SmallDownThoughts',
    'u': 'UpThoughts',
    'i': 'DownThoughts',
    'o': 'BigUpThoughts',
    'p': 'BigDownThoughts',
    'j': 'LeftThoughts',
    'k': 'RightThoughts',
    's': 'StopThoughts',
    '+': 'SpeedUpThoughts',
    '-': 'SlowDownThoughts'
}

COLLECTION_TIME_PER_CLASS = 100  # per class
WINDOW_SLIDE = 0.5

def play_beep():
    winsound.Beep(1000, 400)


def collect_labeled_data(sock_receive, classifier, baseline_mean, baseline_std):
    print("\n🎯 Self-Calibration Mode: Hold a key corresponding to the class.")
    print("Release the key to stop collecting for that class. Be consistent with your mental task.")
    
    # Initialize with balanced collection targets
    labeled_features = {label: [] for label in CLASS_KEYS.values()}
    buffer = deque(maxlen=classifier.samples_per_window + int(WINDOW_SLIDE * classifier.fs))
    
    print("\nPress a key to begin collecting...")
    play_beep()
    
    collection_start = time.time()
    active_class = None
    active_start = None
    
    while True:
        try:
            # Check for timeout (5 minutes max)
            if time.time() - collection_start > 420:
                print("\n🕒 Maximum collection time reached.")
                break
                
            data, _ = sock_receive.recvfrom(classifier.num_channels * 8 * 125)
            eeg_data = np.frombuffer(data, dtype=np.float64).reshape((classifier.num_channels, -1))
            for sample in eeg_data.T:
                buffer.append(sample)
                
            # Check for new key press
            for key, class_label in CLASS_KEYS.items():
                if keyboard.is_pressed(key):
                    if active_class != class_label:
                        active_class = class_label
                        active_start = time.time()
                        print(f"\n🎯 Now collecting: {class_label}")
                        play_beep()
                    break
            else:
                if active_class is not None:
                    print(f"  🛑 Stopped collecting {active_class}")
                    active_class = None
                    active_start = None
                    
            # Process window if we have an active class
            if (active_class is not None and 
                len(buffer) >= classifier.samples_per_window and 
                len(labeled_features[active_class]) < COLLECTION_TIME_PER_CLASS):
                
                window = np.array(buffer)[-classifier.samples_per_window:].T
                window = (window - baseline_mean[:, None]) / (baseline_std[:, None] + 1e-6)
                features = classifier.extract_features(window)
                labeled_features[active_class].append(features)
                
                # Progress feedback
                progress = len(labeled_features[active_class])/COLLECTION_TIME_PER_CLASS*100
                print(f"  {active_class}: {progress:.0f}% complete", end='\r')
                if (progress == 100):
                    play_beep()
                
            # Check if all classes have sufficient data
            if all(len(v) >= COLLECTION_TIME_PER_CLASS for v in labeled_features.values()):
                print("\n✅ Collected sufficient data for all classes")
                break
                            
            if keyboard.is_pressed('esc'):
                print("\n🛑 Exiting self-calibration mode.")
                break
                
        except Exception as e:
            print(f"⚠️ Error during calibration collection: {e}")
            traceback.print_exc()
            continue
            
    return {k: v for k, v in labeled_features.items() if v}


def update_thresholds_from_labeled_data(classifier, labeled_features):
    print("\n🔧 Updating per-class thresholds based on labeled calibration data...")
    
    # Calculate original threshold statistics
    original_thresholds = np.array([classifier.optimal_thresholds[cls] 
                                  for cls in classifier.class_labels])
    mean_orig = np.mean(original_thresholds)
    std_orig = np.std(original_thresholds)
    
    for class_label, feature_list in labeled_features.items():
        if not feature_list or len(feature_list) < 10:  # Minimum samples
            print(f"  ⚠️ Insufficient data ({len(feature_list)} samples) for {class_label}, skipping.")
            continue

        features = np.array(feature_list)
        features_scaled = classifier.scaler.transform(features)
        features_selected = features_scaled[:, classifier.selected_indices]
        
        # Get probabilities for this class
        class_idx = classifier.class_labels.index(class_label)
        proba = classifier.model.predict_proba(features_selected)[:, class_idx]
        
        # Calculate robust threshold (using median and IQR)
        median = np.median(proba)
        q75 = np.percentile(proba, 75)
        q25 = np.percentile(proba, 25)
        iqr = q75 - q25
        new_threshold = max(median - 0.5*iqr, 0)  # Conservative threshold
        
        # Constrain threshold to reasonable range
        orig_thresh = classifier.optimal_thresholds[class_label]
        constrained_thresh = np.clip(new_threshold, 
                                    orig_thresh*0.5, 
                                    orig_thresh*1.5)
        
        print(f"\n{class_label} probabilities: {proba[:5]}")
        print(f"Median: {median:.4f}, IQR: {iqr:.4f}, Raw threshold: {new_threshold:.4f}")

        classifier.optimal_thresholds[class_label] = constrained_thresh
        print(f"  ✅ Updated threshold for {class_label}: {constrained_thresh:.3f} "
              f"(original: {orig_thresh:.3f}, raw: {new_threshold:.3f})")
    
    print("✅ Thresholds updated. Returning to live prediction mode.")


def quick_retrain_with_live_data(classifier, labeled_features):
    print("\n🔄 Quick training using only real-time calibration data...")

    X_new = []
    y_new = []
    for class_label, features in labeled_features.items():
        if len(features) < 5:
            print(f"⚠️ Not enough samples for {class_label}, skipping.")
            continue
        X_new.extend(features)
        y_new.extend([class_label] * len(features))

    if len(set(y_new)) < 2:
        print("❌ Not enough class variety to train a classifier.")
        return

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    X_scaled = classifier.scaler.transform(X_new)
    X_selected = X_scaled[:, classifier.selected_indices]

    model_params = classifier.model.get_params()
    model = RandomForestClassifier(**model_params)
    model.fit(X_selected, y_new)
    classifier.model = model

    optimal_thresholds = {}
    proba = model.predict_proba(X_selected)
    for i, label in enumerate(model.classes_):
        optimal_thresholds[label] = np.percentile(proba[:, i], 25)
    classifier.optimal_thresholds = optimal_thresholds

    print("✅ Model retrained and thresholds updated with real-time data.")

# === Main Execution Block ===
if __name__ == "__main__":
    classifier = EEGMovementClassifier(r"eeg_model_20250407_152309\model_package.joblib")

    UDP_IP = "127.0.0.1"
    UDP_PORT_RECEIVE = 50000
    UDP_PORT_SEND = 60000

    sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_receive.bind((UDP_IP, UDP_PORT_RECEIVE))
    sock_receive.settimeout(1)

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def calibrate_scaler():
        print("\n🧠 Initial Calibration Starting (Close eyes and relax)")
        play_beep()
        time.sleep(3)
        calibration_buffer = []
        start_time = time.time()
        while time.time() - start_time < 20:
            try:
                data, _ = sock_receive.recvfrom(classifier.num_channels * 8 * 125) 
                if data:
                    raw_values = np.frombuffer(data, dtype=np.float64).reshape((classifier.num_channels, -1))
                    calibration_buffer.append(raw_values)
            except socket.timeout:
                continue
        play_beep()
        eeg_cal = np.hstack(calibration_buffer)

        cov = np.cov(eeg_cal)
        d, V = np.linalg.eigh(cov)
        D = np.diag(1. / np.sqrt(d + 1e-6))
        whitening_matrix = np.dot(np.dot(V, D), V.T)
        classifier.whitening_matrix = whitening_matrix

        print(whitening_matrix)

        return np.mean(eeg_cal, axis=1), np.std(eeg_cal, axis=1)

    baseline_mean, baseline_std = calibrate_scaler()

    print("\nPress 'c' within the next 5 seconds to enter Self-Calibration Mode...")
    t_end = time.time() + 5
    while time.time() < t_end:
        if keyboard.is_pressed('c'):
            features = collect_labeled_data(sock_receive, classifier, baseline_mean, baseline_std)
            update_thresholds_from_labeled_data(classifier, features)
            quick_retrain_with_live_data(classifier, features)
            break

    buffer = deque(maxlen=classifier.samples_per_window + int(WINDOW_SLIDE * classifier.fs))
    print("\n🎧 Real-time EEG classification starting...")

    while True:
        try:
            data, _ = sock_receive.recvfrom(classifier.num_channels * 8 * 125)
            eeg_data = np.frombuffer(data, dtype=np.float64).reshape((classifier.num_channels, -1))
            for sample in eeg_data.T:
                buffer.append(sample)

            if len(buffer) >= classifier.samples_per_window:
                window = np.array(buffer)[-classifier.samples_per_window:].T
                window = (window - baseline_mean[:, None]) / (baseline_std[:, None] + 1e-6)
                prediction = classifier.predict(window)
                features = classifier.extract_features(window)
                scaled = classifier.scaler.transform(features.reshape(1, -1))
                selected = scaled[:, classifier.selected_indices]
                proba = classifier.model.predict_proba(selected)[0]
                confidence = max(proba)
                print(f"Predicted: {prediction} | Confidence: {confidence:.3f}")
                sock_send.sendto(prediction.encode(), (UDP_IP, UDP_PORT_SEND))

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            traceback.print_exc()
            continue