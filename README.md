# Interaction Recorder Bot

An automated testing bot that learns from human interactions to perform intelligent website testing.

## Features

- **Interaction Recording**: Records real user interactions with a website using Playwright
- **Advanced ML Model**: Uses a hybrid machine learning approach combining:
  - Random Forest for robust feature learning
  - Neural Network for complex pattern recognition
  - Automatic hyperparameter optimization
- **Automated Testing**: Replays and validates learned interactions
- **HTML Reports**: Generates detailed test reports with GitHub Pages integration
- **Slack Integration**: Automatically notifies about test results via GitHub Actions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Interaction_recorder_bot.git
cd Interaction_recorder_bot
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

### Recording Interactions

Record user interactions with a website:

```bash
python -m src.bot.main
```

Follow the prompts to:
1. Enter the target website URL
2. Specify recording duration
3. Interact with the website naturally

The interactions will be saved to `recorded_interactions.json`.

### Training the Model

Train the hybrid model on recorded interactions:

```bash
python -m src.bot.train_model
```

This will:
1. Load recorded interactions
2. Extract and process features
3. Optimize model hyperparameters
4. Train both Random Forest and Neural Network models
5. Save the hybrid model to `models/hybrid_model.joblib`

### Running Tests

Run automated tests using the trained model:

```bash
python -m src.bot.autonomous_lm_test_2_min
```

The test will:
1. Load the hybrid model
2. Navigate to the target website
3. Perform intelligent interactions based on learned patterns
4. Generate a detailed HTML report
5. Push the report to GitHub Pages
6. Send a notification to Slack (if configured)

## Model Architecture

The hybrid model combines two approaches:

1. **Random Forest Classifier**
   - Handles categorical features
   - Robust to missing data
   - Provides feature importance analysis

2. **Neural Network**
   - Learns complex interaction patterns
   - Better at temporal dependencies
   - Uses dropout for regularization

The predictions are combined using a weighted ensemble approach:
- 60% weight to Random Forest predictions
- 40% weight to Neural Network predictions

## Configuration

### GitHub Pages and Slack Integration

1. Enable GitHub Pages in your repository settings
2. Add a Slack webhook URL to repository secrets:
   - Go to Settings → Secrets and variables → Actions
   - Add `SLACK_WEBHOOK_URL` secret

### Environment Variables

- `GITHUB_PAGES_URL`: URL where test reports are published
- `SLACK_WEBHOOK_URL`: Webhook URL for Slack notifications

## Directory Structure

```
Interaction_recorder_bot/
├── src/
│   └── bot/
│       ├── hybrid_learner.py    # Hybrid ML model implementation
│       ├── executor.py          # Test execution engine
│       ├── recorder.py          # Interaction recorder
│       ├── train_model.py       # Model training script
│       └── main.py             # Main recording script
├── models/                     # Trained model storage
├── reports/                    # Generated test reports
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.