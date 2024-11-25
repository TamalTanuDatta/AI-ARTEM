import json
import os
from src.bot.hybrid_learner import HybridInteractionLearner

def train_model(interactions_file='recorded_interactions.json', model_file='models/hybrid_model.joblib'):
    """Train the hybrid model on recorded interactions."""
    
    # Load recorded interactions
    print(f"\nLoading interactions from {interactions_file}...")
    with open(interactions_file, 'r') as f:
        interactions = json.load(f)
    print(f"Loaded {len(interactions)} interactions")
    
    # Initialize and train the model
    learner = HybridInteractionLearner()
    features, labels = learner.prepare_data(interactions)
    
    print("\nStarting model training...")
    print("This may take a few minutes as we optimize hyperparameters and train both models")
    learner.train(features, labels)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    
    # Save the trained model
    learner.save_model(model_file)
    print(f"\nModel saved to {model_file}")

if __name__ == "__main__":
    train_model()
