import os
import argparse
from src.bot.executor import AutomatedTester

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run autonomous tests')
    parser.add_argument('--duration', type=int, default=120,
                      help='Test duration in seconds (default: 120)')
    parser.add_argument('--url', type=str, default='https://www.leasingmarkt.de',
                      help='Target website URL')
    args = parser.parse_args()
    
    # Use the hybrid model path
    model_path = 'models/hybrid_model.joblib'
    tester = AutomatedTester(model_path, 'recorded_interactions.json')
    report = tester.run_tests(args.url, duration=args.duration)
    
    if report:
        report.generate_html_report()

if __name__ == "__main__":
    main()
