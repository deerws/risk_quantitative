# src/agents/fine_tuning.py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import optuna

class FineTuningEngine:
    """Motor de fine-tuning para modelos dos agents"""
    
    def __init__(self):
        self.study = None
        
    def optimize_kmeans_clusters(self, features: pd.DataFrame, max_clusters: int = 10):
        """Otimiza número de clusters usando método do cotovelo + silhouette"""
        from sklearn.metrics import silhouette_score
        
        best_k = 3
        best_score = -1
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        return best_k, best_score
    
    def optimize_random_forest(self, X, y):
        """Otimiza hiperparâmetros do Random Forest"""
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_params_, grid_search.best_score_