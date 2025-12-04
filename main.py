
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Force TensorFlow to use the CPU if CUDA fails
tf.config.set_visible_devices([], 'GPU')



# Main script
if __name__ == "__main__":
    # Load Data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X_train = train.drop(columns=['subject', 'Activity'])
    y_train = train['Activity']
    X_test = test.drop(columns=['subject', 'Activity'])
    y_test = test['Activity']

    # Number of classes
    num_classes = y_train.nunique()

    # Initialize Components
    feature_size = X_train.shape[1] * 4
    segmenter = AdaptiveActivitySegmenter(window_size=15, base_threshold=0.3, feature_size=feature_size)
    enhanced_model = EnhancedActivityRecognitionModel(segmenter, lstm_units=128,
                                                      probability_threshold=0.4, num_classes=num_classes)

    # Train
    enhanced_model.fit(X_train, y_train, epochs=30, batch_size=64)

    # Predict
    predictions = enhanced_model.predict(X_test)

    # Map y_test to encoded values for comparison
    encoder = LabelEncoder()
    y_test_encoded = encoder.fit_transform(y_test)

    print(classification_report(y_test_encoded, predictions))
    
    # Plot confusion matrix
    plt.figure(dpi=250, figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_encoded, predictions)) 
    disp.plot(cmap='inferno')
    plt.title("Confusion matrix of results from Improved Model")
    plt.xlabel("Predicted Activity")
    plt.ylabel("True Activity")
    plt.savefig("final_mod.png")
    plt.show()
    
