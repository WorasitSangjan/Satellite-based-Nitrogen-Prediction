import platform
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
import keras_tuner as kt

def setup_gpu():
    """Configure GPU based on platform"""
    system = platform.system()
    print(f"Running on {system} platform")
    
    try:
        if system == "Darwin":  # macOS
            # Configure Metal GPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"Metal GPU acceleration enabled. Found {len(physical_devices)} GPU(s)")
                # Optimize thread usage for Metal
                tf.config.threading.set_intra_op_parallelism_threads(4)
                tf.config.threading.set_inter_op_parallelism_threads(4)
                return "metal"
                
        elif system == "Windows":  # Windows
            # Configure CUDA GPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"CUDA GPU acceleration enabled. Found {len(physical_devices)} GPU(s)")
                return "cuda"
        
        # If no GPU found or on different platform
        print("No GPU devices found or unsupported platform. Running on CPU")
        return "cpu"
        
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        print("Falling back to CPU")
        return "cpu"

class NitrogenModel(kt.HyperModel):
    def __init__(self, input_dim, gpu_type="cpu"):
        self.input_dim = input_dim
        self.gpu_type = gpu_type
    
    def build(self, hp):
        model = Sequential()
        
        # First layer - keep it simple but well-regularized
        model.add(Dense(
            units=32,
            activation='relu',
            input_dim=self.input_dim,
            kernel_regularizer=l1_l2(
                l1=hp.Float('l1', 1e-5, 1e-3, sampling='log'),
                l2=hp.Float('l2', 1e-5, 1e-3, sampling='log')
            )
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', 0.3, 0.5, step=0.1)))
        
        # Single hidden layer
        model.add(Dense(
            units=16,
            activation='relu',
            kernel_regularizer=l1_l2(
                l1=hp.Float('l1_2', 1e-5, 1e-3, sampling='log'),
                l2=hp.Float('l2_2', 1e-5, 1e-3, sampling='log')
            )
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', 0.2, 0.4, step=0.1)))
        
        # Output layer
        model.add(Dense(1))
        
        # Optimizer configuration
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

def train_model(data_path='trainingM10.csv'):
    # Setup GPU configuration
    gpu_type = setup_gpu()
    
    print("Loading data...")
    data = pd.read_csv(data_path)
    X = data.drop('Total_N', axis=1)
    y = data['Total_N']
    
    X = X.values
    y = y.values
    feature_names = data.drop('Total_N', axis=1).columns.tolist()
    
    cv_scores = {'r2': [], 'rmse': [], 'mae': []}
    best_score = -np.inf
    best_model = None
    best_scaler = None
    
    # 3-fold CV
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    # Determine batch size based on GPU type
    batch_size = 32 if gpu_type == "metal" else 16
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/3")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        tuner = kt.Hyperband(
            NitrogenModel(X.shape[1], gpu_type),
            objective='val_loss',
            max_epochs=50,
            factor=3,
            directory='nitrogen_tuning',
            project_name=f'fold_{fold}'
        )
        
        tuner.search(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            callbacks=callbacks,
            epochs=50,
            batch_size=batch_size,
            verbose=1
        )
        
        best_hps = tuner.get_best_hyperparameters(1)[0]
        model = tuner.hypermodel.build(best_hps)
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=75,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        cv_scores['r2'].append(r2)
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        
        # Keep track of best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_scaler = scaler
        
        print(f"\nFold {fold + 1} Results:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    # Calculate final scores
    final_scores = {
        metric: {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        for metric, scores in cv_scores.items()
    }
    
    print("\nFinal Cross-validation Results:")
    for metric, values in final_scores.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Save the best model and scaler
    print("\nSaving best model and scaler...")
    
    # Save model
    best_model.save('nitrogen_model')
    
    # Save scaler and feature names
    with open('model_info.pkl', 'wb') as f:
        pickle.dump({
            'scaler': best_scaler,
            'feature_names': feature_names,
            'cv_scores': final_scores
        }, f)
    
    print("Model and associated files saved:")
    print("- Model: 'nitrogen_model/'")
    print("- Scaler and info: 'model_info.pkl'")
    
    return final_scores, best_model, best_scaler

def load_and_predict(X_new, model_path='nitrogen_model', info_path='model_info.pkl'):
    """Function to load model and make predictions"""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load scaler and feature names
    with open(info_path, 'rb') as f:
        model_info = pickle.load(f)
    
    scaler = model_info['scaler']
    feature_names = model_info['feature_names']
    
    # Ensure X_new has correct features in correct order
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new[feature_names]
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    
    return predictions

if __name__ == "__main__":
    scores, best_model, best_scaler = train_model()
    
    # Example prediction
    print("\nExample prediction with the saved model:")
    data = pd.read_csv('trainingM10.csv')
    X_example = data.drop('Total_N', axis=1).head(1)
    pred = load_and_predict(X_example)
    print(f"Prediction for first sample: {pred[0][0]:.2f}")