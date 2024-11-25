import os
from src.bot.executor import AutomatedTester

def main():
    # Configuration
    github_url = os.getenv('GITHUB_PAGES_URL')  # e.g., "https://username.github.io/repo/reports/index.html"
    
    # Use the new hybrid model path
    model_path = 'models/hybrid_model.joblib'
    tester = AutomatedTester(model_path, 'recorded_interactions.json')
    report = tester.run_tests('https://www.leasingmarkt.de', duration=120)
    
    # Generate report
    if report:
        report.generate_html_report()
        if github_url:
            print("\nNote: To enable report publishing, set these environment variables:")
            print("- GITHUB_PAGES_URL: URL where the report will be published")

if __name__ == "__main__":
    main()
