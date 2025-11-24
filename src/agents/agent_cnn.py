# src/agents/agent_cnn.py
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

class AgentCNN:
    """Detecção de padrões técnicos usando CNN 1D"""
    
    def __init__(self):
        self.model = None
        self.patterns = ['head_shoulders', 'double_top', 'triangle', 'channel']
    
    def build_cnn_model(self, input_shape, num_classes):
        """Constrói CNN para reconhecimento de padrões"""
        model = tf.keras.Sequential([
            Conv1D(32, 5, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(64, 5, activation='relu'),
            MaxPooling1D(2),
            Conv1D(128, 5, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def detect_chart_patterns(self, price_series, window=20):
        """Detecta padrões gráficos em séries de preços"""
        # Criar subsequências (sliding window)
        sequences = []
        for i in range(len(price_series) - window):
            seq = price_series[i:i+window]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        
        # Aqui precisaríamos de dados rotulados para treinar
        # Por enquanto, retornar estrutura básica
        return {
            'pattern_detection': 'CNN implementada - necessita dados rotulados',
            'sequences_created': len(sequences),
            'window_size': window
        }