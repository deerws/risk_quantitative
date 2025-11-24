# src/agents/agent_autoencoder.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

class AgentAutoencoder:
    """Detecção de anomalias usando Autoencoders"""
    
    def __init__(self, encoding_dim=32):
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.threshold = None
    
    def build_autoencoder(self, input_dim):
        """Constrói autoencoder para detecção de anomalias"""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def detect_anomalies(self, market_features):
        """Detecta anomalias usando reconstruction error"""
        if self.autoencoder is None:
            self.autoencoder = self.build_autoencoder(market_features.shape[1])
        
        # Normalizar dados
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(market_features)
        
        # Treinar autoencoder
        history = self.autoencoder.fit(
            scaled_data, scaled_data,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            verbose=0
        )
        
        # Calcular reconstruction error
        reconstructions = self.autoencoder.predict(scaled_data)
        mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
        
        # Definir threshold (percentil 95%)
        self.threshold = np.percentile(mse, 95)
        anomalies = mse > self.threshold
        
        return {
            'anomaly_indices': np.where(anomalies)[0],
            'reconstruction_error': mse,
            'threshold': self.threshold,
            'training_loss': history.history['loss'][-1]
        }