import json
from src.bot.recorder import InteractionRecorder
from src.bot.learner import InteractionLearner

def main():
    # Configuration
    URL = "https://www.leasingmarkt.de"
    RECORDING_FILE = "src/bot/recorded_interactions.json"
    MODEL_FILE = "src/bot/interaction_model.joblib"
    RECORDING_DURATION = 120  # 2 minutes of recording time
    
    try:
        # Step 1: Record user interactions
        print(f"\nWill record interactions for: {URL}")
        recorder = InteractionRecorder()
        recorder.start_recording(URL, duration=RECORDING_DURATION)
        
        # Save recorded interactions
        print(f"\nSaving recorded interactions to {RECORDING_FILE}...")
        recorder.save_interactions(RECORDING_FILE)
        print("Interactions saved successfully!")
        
        # Step 2: Train the model
        print("\nPreparing to train model...")
        with open(RECORDING_FILE, 'r') as f:
            interactions = json.load(f)
            
        if not interactions:
            print("No interactions were recorded. Please try again.")
            return
            
        print(f"Loaded {len(interactions)} recorded interactions.")
        
        # Train and save the model
        learner = InteractionLearner()
        features, labels = learner.prepare_data(interactions)
        learner.train(features, labels)
        
        print(f"\nSaving trained model to {MODEL_FILE}...")
        learner.save_model(MODEL_FILE)
        print("Model saved successfully!")
        
        print("\nRecording and training completed! You can now use the trained model for testing.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try again. If the error persists, check your internet connection.")

if __name__ == "__main__":
    main()