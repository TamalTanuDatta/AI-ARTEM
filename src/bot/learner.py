import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class InteractionLearner:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, interactions):
        # Convert interaction data to features
        features = []
        labels = []
        
        for interaction in interactions:
            # Extract relevant features from interactions
            feature_vector = self._extract_features(interaction)
            features.append(feature_vector)
            labels.append(interaction['type'])
            
        # Encode labels and train model
        encoded_labels = self.label_encoder.fit_transform(labels)
        return np.array(features), encoded_labels
        
    def train(self, features, labels):
        print("\nTraining model on recorded interactions...")
        try:
            self.model.fit(features, labels)
            print("Model training completed successfully!")
            
            # Print some basic model information
            feature_importance = self.model.feature_importances_
            print("\nTop important features:")
            for i, importance in enumerate(feature_importance):
                if importance > 0.1:  # Only show significant features
                    print(f"Feature {i}: {importance:.3f}")
                    
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise
        
    def save_model(self, filename):
        joblib.dump({
            'model': self.model,
            'encoder': self.label_encoder
        }, filename)
        
    def _extract_features(self, interaction):
        # Convert interaction properties into numerical features
        features = []
        
        # Type encoding
        event_types = ['click', 'input', 'navigation', 'keydown', 'mousemove']
        features.extend([1 if interaction['type'] == t else 0 for t in event_types])
        
        # Selector hash - for element identification
        features.append(hash(interaction.get('selector', '')) % 1e6)  # Modulo to prevent overflow
        
        # Value length for input events
        features.append(len(interaction.get('value', '')))
        
        # Position features for click and mousemove
        position = interaction.get('position', {'x': 0, 'y': 0})
        if position:
            features.extend([position['x'], position['y']])
        else:
            features.extend([0, 0])
        
        # URL hash
        features.append(hash(interaction.get('url', '')) % 1e6)
        
        return features