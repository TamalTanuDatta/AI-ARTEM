# Autonomous Testing Bot

This project implements an autonomous testing bot that learns from user interactions and performs automated testing using machine learning and Playwright.

## Architecture Overview

The system consists of three main components:

1. **Interaction Recorder**: Records user interactions with the website
2. **Interaction Learner**: Trains a machine learning model on recorded interactions
3. **Automated Tester**: Executes automated tests based on learned patterns

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers:
```bash
playwright install
```

## Usage

1. Run the main script:
```bash
python src/bot/main.py
```

2. The system will:
   - Record your interactions with the website
   - Train a model based on your interactions
   - Execute automated tests using the learned patterns

## How to Run Tests

You can run the autonomous testing bot for 2 minutes with this command:

```bash
python3 -m src.bot.autonomous_lm_test_2_min
```

### Test Reports
- The test results will be saved as an HTML file in the `reports` directory
- Each time you run the tests, the previous report will be automatically deleted to maintain cleanliness
- The bot performs various assertions during test execution to validate website behavior and functionality

### What the Bot Tests
The autonomous testing bot:
- Navigates through the website
- Interacts with various UI elements
- Validates links and page responses
- Performs assertions on page content and functionality
- Generates comprehensive test reports with details of all actions and validations

## Nightly Automated Tests

The repository runs autonomous tests nightly at 00:00 UTC using GitHub Actions. Test reports are:
- Generated as HTML files
- Published to GitHub Pages
- Available as downloadable artifacts
- Kept for 14 days

### Viewing Test Reports
- Latest report: Visit `https://TamalTanuDatta.github.io/LM_Autonomous_Testing_Bot/`
- Historical reports: Download from GitHub Actions workflow runs

### Manual Trigger
1. Go to Actions tab
2. Select "Nightly Autonomous Tests"
3. Click "Run workflow"

## Test Reports

The automated test reports are available in two locations:

1. **GitHub Pages (Latest Report)**
   - Visit [https://tamaltanudatta.github.io/LM_Autonomous_Testing_Bot/](https://tamaltanudatta.github.io/LM_Autonomous_Testing_Bot/)
   - This page is automatically updated after each test run

2. **GitHub Actions Artifacts**
   - Go to the [Actions tab](https://github.com/TamalTanuDatta/LM_Autonomous_Testing_Bot/actions)
   - Click on any workflow run
   - Scroll to the "Artifacts" section
   - Download the "test-report" artifact

Reports include:
- Test execution timeline
- Successful and failed actions
- Screenshots of failures (if any)
- Interaction statistics
- Coverage metrics

## Components

### InteractionRecorder
- Records user clicks, inputs, and navigation
- Saves interaction data in JSON format

### InteractionLearner
- Processes recorded interactions
- Trains a Random Forest model
- Converts interaction patterns into features

### AutomatedTester
- Loads trained model
- Analyzes webpage elements
- Executes predicted actions

## Customization

- Modify feature extraction in both learner and executor
- Adjust sleep timings for different websites
- Add more interaction types as needed