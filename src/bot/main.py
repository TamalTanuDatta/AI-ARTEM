import argparse
from datetime import datetime
import json
import os

from .executor import InteractionExecutor
from .hybrid_learner import HybridInteractionLearner
from .recorder import InteractionRecorder


DEFAULT_URL = "https://www.leasingmarkt.de"


def main(url=DEFAULT_URL, recording_duration=240, train_only=False):
    """Main function to record interactions, train the model, and execute learned behaviors.

    Args:
        url (str): The URL to record interactions from (default: www.leasingmarkt.de)
        recording_duration (int): Duration in seconds to record interactions
        train_only (bool): If True, only train the model using existing interactions
    """
    print("\n=== Starting Autonomous Web Testing Bot ===\n")

    # Initialize components
    recorder = InteractionRecorder()
    learner = HybridInteractionLearner()
    executor = InteractionExecutor()

    # Create data directory if it doesn't exist
    os.makedirs("src/bot/data", exist_ok=True)

    try:
        if not train_only:
            # Step 1: Record Interactions
            print(f"Starting interaction recording for {recording_duration} seconds on {url}")
            print("Please interact with the website when the browser opens...")

            start_time = datetime.now()
            interactions = recorder.start_recording(url, duration=recording_duration)

            if not interactions:
                print("No interactions recorded. Please try again and interact with the website.")
                return

            # Save interactions to file
            interaction_file = "src/bot/data/recorded_interactions.json"
            print(f"\nRecorded {len(interactions)} interactions")
            print(f"Saving interactions to {interaction_file}")

            with open(interaction_file, "w") as f:
                json.dump(interactions, f, indent=2)
        else:
            # Load existing interactions for training
            interaction_file = "src/bot/data/recorded_interactions.json"
            print(f"Loading existing interactions from {interaction_file}")
            try:
                with open(interaction_file, "r") as f:
                    interactions = json.load(f)
                print(f"Loaded {len(interactions)} interactions")
            except FileNotFoundError:
                print(f"Error: {interaction_file} not found. Please record interactions first.")
                return

        # Step 2: Prepare Data and Optimize
        print("\nPreparing data and optimizing hyperparameters...")
        features, labels = learner.prepare_data(interactions)

        print("\nStarting hyperparameter optimization...")
        print("This may take a few minutes...")
        best_params = learner.optimize_hyperparameters(features, labels, n_trials=20)

        print("\nBest hyperparameters found:")
        for param, value in best_params.items():
            print(f"- {param}: {value}")

        # Step 3: Train Model
        print("\nTraining model with optimized parameters...")
        learner.train(features, labels)

        # Step 4: Execute Learned Behaviors
        print("\nExecuting learned interactions...")
        executor.execute_interactions(learner)

        # Print summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() if not train_only else (end_time - datetime.now()).total_seconds()

        print("\n=== Process Summary ===")
        print(f"Total interactions recorded: {len(interactions)}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Interaction data saved to: {interaction_file}" if not train_only else f"Interaction data loaded from: {interaction_file}")
        print("\nProcess completed successfully!")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record and learn web interactions.')
    parser.add_argument('--url', default=DEFAULT_URL, help=f'The URL to record interactions from (default: {DEFAULT_URL})')
    parser.add_argument('--duration', type=int, default=240, help='Duration in seconds to record interactions (default: 240)')
    parser.add_argument('--train-only', action='store_true', help='Only train the model using existing recorded interactions')

    args = parser.parse_args()
    main(args.url, args.duration, args.train_only)