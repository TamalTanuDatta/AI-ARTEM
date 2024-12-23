from .executor import InteractionExecutor 

from .hybrid_learner import HybridInteractionLearner 

import json 

 

def main(): 

    """ 

    Run autonomous tests using the trained model and recorded interactions. 

    Duration: 3 minutes (180 seconds) 

    """ 

    print("\n=== Starting Autonomous Testing ===") 

     

    # Load recorded interactions for reference 

    try: 

        with open('src/bot/data/recorded_interactions.json', 'r') as f: 

            recorded_interactions = json.load(f) 

        print(f"Loaded {len(recorded_interactions)} recorded interactions for reference") 

    except Exception as e: 

        print(f"Warning: Could not load recorded interactions: {str(e)}") 

        recorded_interactions = [] 

 

    # Initialize components 

    learner = HybridInteractionLearner() 

    executor = InteractionExecutor() 

     

    print("\nStarting automated test execution...") 

    print("Duration: 3 minutes") 

    print("Target URL: https://www.leasingmarkt.de") 

     

    # Execute learned behaviors 

    executor.execute_interactions(learner, duration=180) 

     

    print("\n=== Testing Completed ===") 

 

if __name__ == "__main__": 

    main() 