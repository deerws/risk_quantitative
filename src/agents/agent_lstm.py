# src/agents/agent_lstm.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class AgentLSTM:
    """Agente para previsão de séries temporais com LSTM"""
    
    def __init__(self):
        self.model = None
        self.lookback = 60  # 60 dias de histórico
        self.forecast_days = 10  # Prever 10 dias à frente
    
    def build_lstm_model(self, input_shape):
        """Constrói modelo LSTM para previsão"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(self.forecast_days)  # Prever múltiplos dias
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def predict_volatility_regime(self, returns_series):
        """Prevê regime de volatilidade usando LSTM"""
        # Preparar dados sequenciais
        sequences = self.create_sequences(returns_series, self.lookback)
        
        if len(sequences) < 100:
            return None
        
        # Treinar modelo (usando dados históricos)
        X_train, y_train = self.prepare_training_data(sequences)
        
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Treinar com early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Fazer previsões
        future_predictions = self.model.predict(sequences[-100:])
        
        return {
            'predictions': future_predictions,
            'training_loss': history.history['loss'][-1],
            'model_architecture': 'LSTM_3Layer_Dropout'
        }