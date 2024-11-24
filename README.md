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