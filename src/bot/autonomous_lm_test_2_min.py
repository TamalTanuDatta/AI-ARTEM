from src.bot.executor import AutomatedTester

def main():
    tester = AutomatedTester('interaction_model.joblib', 'recorded_interactions.json')
    tester.run_tests('https://www.leasingmarkt.de', duration=120)

if __name__ == "__main__":
    main()
