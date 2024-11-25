import json
import os
from src.bot.hybrid_learner import HybridInteractionLearner
from src.bot.recorder import InteractionRecorder

# Constants
MODEL_FILE = 'models/hybrid_model.joblib'
INTERACTIONS_FILE = 'recorded_interactions.json'

def record_interactions(url, duration=300):
    """Record user interactions on a website."""
    recorder = InteractionRecorder()
    
    try:
        print(f"\nStarting interaction recording at {url}")
        print(f"Recording will run for {duration} seconds")
        print("Please interact with the website naturally...")
        
        interactions = recorder.record(url, duration)
        
        # Save recorded interactions
        with open(INTERACTIONS_FILE, 'w') as f:
            json.dump(interactions, f, indent=2)
            
        print(f"\nRecorded {len(interactions)} interactions")
        print(f"Interactions saved to {INTERACTIONS_FILE}")
        
        return interactions
        
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        return None

def train_model(interactions):
    """Train the model on recorded interactions."""
    try:
        learner = HybridInteractionLearner()
        features, labels = learner.prepare_data(interactions)
        learner.train(features, labels)
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        learner.save_model(MODEL_FILE)
        
        print(f"\nModel saved to {MODEL_FILE}")
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

def main():
    # Record interactions
    url = input("\nEnter the URL to record interactions from: ")
    duration = int(input("Enter recording duration in seconds (default: 300): ") or "300")
    
    interactions = record_interactions(url, duration)
    
    if interactions:
        # Train model
        print("\nTraining model on recorded interactions...")
        if train_model(interactions):
            print("\nProcess completed successfully!")
        else:
            print("\nFailed to train model.")
    else:
        print("\nNo interactions recorded. Process aborted.")

if __name__ == "__main__":
    main()