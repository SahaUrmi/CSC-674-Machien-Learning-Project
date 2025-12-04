# EnhancedActivityRecognitionModel
class EnhancedActivityRecognitionModel:
    def __init__(self, segmenter, lstm_units=128, probability_threshold=0.5, num_classes=6):
        self.segmenter = segmenter
        self.probability_threshold = probability_threshold
        self.num_classes = num_classes
        self.activity_log = []
        self.scaler = StandardScaler()

        # LSTM model
        self.lstm_model = Sequential([
            Input(shape=(None, segmenter.feature_size)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(lstm_units // 2, return_sequences=False),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

    def fit(self, X, y, epochs=10, batch_size=32):
        X_scaled = self.scaler.fit_transform(X)

        # Label encoding (no one-hot encoding)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Create rolling window features
        X_rolling = []
        for i in range(len(X_scaled)):
            rolling_features = self._extract_rolling_features(X_scaled, i)
            self.segmenter.detect_change_point(rolling_features)
            X_rolling.append(rolling_features)

        # Shape for LSTM input: (samples, timesteps, feature_size)
        X_rolling = np.array(X_rolling).reshape(len(X_rolling), 1, self.segmenter.feature_size)

        # Train the LSTM model
        self.lstm_model.fit(X_rolling, y_encoded, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        predictions = []
        X_scaled = self.scaler.transform(X)
        X_rolling = []

        for i in range(len(X_scaled)):
            rolling_features = self._extract_rolling_features(X_scaled, i)
            self.segmenter.detect_change_point(rolling_features)
            X_rolling.append(rolling_features)

        # Predict using LSTM
        X_rolling = np.array(X_rolling).reshape(len(X_rolling), 1, self.segmenter.feature_size)
        lstm_predictions = self.lstm_model.predict(X_rolling)

        # Final predictions based on probabilities
        for lstm_pred in lstm_predictions:
            if lstm_pred.max() > self.probability_threshold:
                predictions.append(np.argmax(lstm_pred))
            else:
                predictions.append(-1)  # Unknown class
        return np.array(predictions)

    def _extract_rolling_features(self, X, idx):
        window_size = min(5, idx + 1)
        rolling_window = X[idx - window_size + 1: idx + 1]
        features = np.hstack([
            rolling_window.mean(axis=0),
            rolling_window.std(axis=0),
            rolling_window.min(axis=0),
            rolling_window.max(axis=0)
        ])
        return features
