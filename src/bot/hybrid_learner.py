import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna


class InteractionNet(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_size=64, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Deep learning layers with dropout for regularization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        self.learning_rate = learning_rate

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc3(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class HybridInteractionLearner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.input_size = None
        self.num_classes = None

    def prepare_data(self, interactions):
        """
        Convert interaction data to features and labels for training.

        Args:
            interactions (list): List of recorded interactions

        Returns:
            tuple: (features, labels) preprocessed for training
        """
        features = []
        labels = []

        for interaction in interactions:
            feature_vector = self._extract_features(interaction)
            features.append(feature_vector)
            labels.append(interaction['type'])

        features = np.array(features, dtype=np.float32)
        features = self.scaler.fit_transform(features)

        encoded_labels = self.label_encoder.fit_transform(labels)

        self.input_size = features.shape[1]
        self.num_classes = len(self.label_encoder.classes_)

        return features, encoded_labels

    def optimize_hyperparameters(self, features, labels, n_trials=50):
        def objective(trial):
            hidden_size = trial.suggest_int('hidden_size', 32, 256)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

            model = InteractionNet(
                input_size=self.input_size,
                num_classes=self.num_classes,
                hidden_size=hidden_size,
                learning_rate=learning_rate
            )

            train_size = int(0.8 * len(features))
            X_train, X_val = features[:train_size], features[train_size:]
            y_train, y_val = labels[:train_size], labels[train_size:]

            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )

            trainer = pl.Trainer(
                max_epochs=10,
                enable_progress_bar=False,
                logger=False
            )

            trainer.fit(
                model,
                torch.utils.data.DataLoader(train_dataset, batch_size=32),
                torch.utils.data.DataLoader(val_dataset, batch_size=32)
            )

            return trainer.callback_metrics['val_loss'].item()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def train(self, features, labels):
        print("\nTraining hybrid model on recorded interactions...")
        try:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(features),
                torch.LongTensor(labels)
            )

            self.model = InteractionNet(
                input_size=self.input_size,
                num_classes=self.num_classes
            )

            trainer = pl.Trainer(
                max_epochs=100,
                enable_progress_bar=True,
                logger=True
            )

            trainer.fit(
                self.model,
                torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            )

            print("Model training completed successfully!")

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise

    def predict(self, interaction):
        features = np.array([self._extract_features(interaction)])
        features = self.scaler.transform(features)

        features_tensor = torch.FloatTensor(features)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = torch.exp(logits)
            predicted_class = torch.argmax(probabilities, dim=1)

        return self.label_encoder.inverse_transform(predicted_class.numpy())

    def _extract_features(self, interaction):
        """
        Extract numerical features from an interaction.

        Args:
            interaction (dict): The interaction data

        Returns:
            list: Numerical features for model training
        """
        features = []

        features.extend([
            float(interaction.get('x', 0)),
            float(interaction.get('y', 0))
        ])

        url = interaction.get('url', '')
        features.append(float(len(url)))

        selector = interaction.get('selector', '')
        features.append(float(len(selector)))

        value = interaction.get('value', '')
        features.append(float(len(str(value))))

        return features