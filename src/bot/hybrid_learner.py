import numpy as np
import torch
import torch.nn as nn
import lightning as L
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import optuna
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
import os
import joblib

class InteractionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class InteractionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class InteractionModule(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = InteractionNet(input_size, hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class HybridInteractionLearner:
    def __init__(self):
        self.rf_model = RandomForestClassifier()
        self.dl_model = None
        self.label_encoder = LabelEncoder()
        self.best_params = None
        self.feature_columns = None
        
    def _extract_features(self, interaction):
        """Extract features from an interaction with security checks."""
        features = []
        
        try:
            # Validate interaction data
            if not isinstance(interaction, dict):
                raise ValueError("Invalid interaction format")
            
            # Event type encoding with validation
            event_types = ['click', 'input', 'navigation', 'keydown', 'mousemove']
            event_type = interaction.get('type', '').lower()
            features.extend([1 if event_type == t else 0 for t in event_types])
            
            # Element features with sanitization
            selector = str(interaction.get('selector', ''))[:1000]  # Limit length
            selector_hash = hash(selector) % (2**32)  # Use 32-bit hash
            features.append(selector_hash)
            features.append(min(len(selector), 1000))  # Cap length
            
            # Value features with sanitization
            value = str(interaction.get('value', ''))[:100]  # Limit length
            features.append(min(len(value), 100))
            features.append(1 if value.isdigit() else 0)
            features.append(1 if '@' in value and '.' in value else 0)
            
            # Position features with bounds checking
            position = interaction.get('position', {'x': 0, 'y': 0})
            x = max(min(float(position.get('x', 0)), 10000), 0)  # Bound coordinates
            y = max(min(float(position.get('y', 0)), 10000), 0)
            features.extend([x, y, 1 if x > 0 and y > 0 else 0])
            
            # Temporal features with validation
            try:
                timestamp = datetime.fromisoformat(interaction.get('timestamp', datetime.now().isoformat()))
            except (ValueError, TypeError):
                timestamp = datetime.now()
            
            features.extend([
                timestamp.hour,
                timestamp.minute,
                1 if 9 <= timestamp.hour <= 17 else 0
            ])
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            # Return safe default features
            return [0] * 15  # Match feature length
        
    def prepare_data(self, interactions):
        """Convert interactions to features and labels."""
        features = []
        labels = []
        
        for interaction in interactions:
            feature_vector = self._extract_features(interaction)
            features.append(feature_vector)
            labels.append(interaction['type'])
            
        # Create feature names
        if self.feature_columns is None:
            self.feature_columns = [
                'type_click', 'type_input', 'type_navigation', 'type_keydown', 'type_mousemove',
                'selector_hash', 'selector_length',
                'value_length', 'is_numeric', 'is_email',
                'pos_x', 'pos_y', 'is_visible',
                'hour', 'minute', 'business_hours'
            ]
        
        # Convert to numpy arrays
        features = np.array(features)
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        return features, encoded_labels
        
    def optimize_hyperparameters(self, features, labels, n_trials=50):
        """Use Optuna to find the best hyperparameters."""
        def objective(trial):
            # Random Forest parameters
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('rf_max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
            }
            
            # Neural Network parameters
            nn_params = {
                'hidden_size': trial.suggest_int('nn_hidden_size', 32, 256),
                'learning_rate': trial.suggest_float('nn_learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 128)
            }
            
            # Train Random Forest
            rf = RandomForestClassifier(**rf_params)
            rf.fit(features, labels)
            rf_score = rf.score(features, labels)
            
            # Train Neural Network
            dataset = InteractionDataset(features, labels)
            train_loader = DataLoader(dataset, batch_size=nn_params['batch_size'], shuffle=True)
            
            model = InteractionModule(
                input_size=features.shape[1],
                hidden_size=nn_params['hidden_size'],
                num_classes=len(np.unique(labels)),
                learning_rate=nn_params['learning_rate']
            )
            
            trainer = L.Trainer(
                max_epochs=5,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False
            )
            
            trainer.fit(model, train_loader)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(features))
                nn_preds = torch.argmax(logits, dim=1).numpy()
                nn_score = (nn_preds == labels).mean()
            
            # Combine scores
            return 0.6 * rf_score + 0.4 * nn_score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return study.best_value
        
    def train(self, features, labels):
        """Train both Random Forest and Neural Network models."""
        print("\nTraining hybrid model on recorded interactions...")
        
        try:
            if self.best_params is None:
                print("Optimizing hyperparameters...")
                self.optimize_hyperparameters(features, labels)
            
            # Train Random Forest
            rf_params = {
                'n_estimators': self.best_params['rf_n_estimators'],
                'max_depth': self.best_params['rf_max_depth'],
                'min_samples_split': self.best_params['rf_min_samples_split']
            }
            self.rf_model = RandomForestClassifier(**rf_params)
            self.rf_model.fit(features, labels)
            
            # Train Neural Network
            dataset = InteractionDataset(features, labels)
            train_loader = DataLoader(
                dataset,
                batch_size=self.best_params['batch_size'],
                shuffle=True
            )
            
            self.dl_model = InteractionModule(
                input_size=features.shape[1],
                hidden_size=self.best_params['nn_hidden_size'],
                num_classes=len(np.unique(labels)),
                learning_rate=self.best_params['nn_learning_rate']
            )
            
            trainer = L.Trainer(
                max_epochs=10,
                enable_progress_bar=True,
                enable_model_summary=True
            )
            
            trainer.fit(self.dl_model, train_loader)
            
            print("\nModel training completed successfully!")
            self._print_feature_importance()
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise
            
    def _print_feature_importance(self):
        """Print feature importance from Random Forest model."""
        if self.rf_model and self.feature_columns:
            importance = self.rf_model.feature_importances_
            feat_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop important features:")
            for _, row in feat_imp.head().iterrows():
                print(f"• {row['feature']}: {row['importance']:.3f}")
    
    def predict(self, features):
        """Ensemble prediction combining both models."""
        if self.rf_model is None or self.dl_model is None:
            raise ValueError("Models not trained. Call train() first.")
            
        # Get Random Forest predictions
        rf_pred = self.rf_model.predict_proba(features)
        
        # Get Neural Network predictions
        self.dl_model.eval()
        with torch.no_grad():
            logits = self.dl_model(torch.FloatTensor(features))
            dl_pred = torch.softmax(logits, dim=1).numpy()
        
        # Weighted average of predictions (60% RF, 40% NN)
        ensemble_pred = 0.6 * rf_pred + 0.4 * dl_pred
        return np.argmax(ensemble_pred, axis=1)
    
    def save_model(self, filename):
        """Save both models and metadata."""
        model_dir = os.path.dirname(filename)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Random Forest and metadata
        joblib.dump({
            'rf_model': self.rf_model,
            'label_encoder': self.label_encoder,
            'best_params': self.best_params,
            'feature_columns': self.feature_columns
        }, filename)
        
        # Save Neural Network
        if self.dl_model:
            torch.save(self.dl_model.state_dict(), f"{filename}_nn.pt")
        
    def load_model(self, filename):
        """Load both models and metadata."""
        # Load Random Forest and metadata
        data = joblib.load(filename)
        self.rf_model = data['rf_model']
        self.label_encoder = data['label_encoder']
        self.best_params = data['best_params']
        self.feature_columns = data['feature_columns']
        
        # Load Neural Network if it exists
        nn_path = f"{filename}_nn.pt"
        if os.path.exists(nn_path):
            self.dl_model = InteractionModule(
                input_size=len(self.feature_columns),
                hidden_size=self.best_params['nn_hidden_size'],
                num_classes=len(self.label_encoder.classes_),
                learning_rate=self.best_params['nn_learning_rate']
            )
            self.dl_model.load_state_dict(torch.load(nn_path))
