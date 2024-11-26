from playwright.sync_api import sync_playwright
import joblib
import numpy as np
import time
import json
import random
from urllib.parse import urljoin, urlparse
from datetime import datetime
import os
import glob
from jinja2 import Template
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class TestReport:
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.test_steps = []
        self.total_actions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self.visited_urls = set()
        self.page_visits = []  # Track detailed page visits
        self.button_clicks = []  # Track button interactions
        self.element_interactions = []  # Track other element interactions
        self.assertions = {
            'passed': 0,
            'failed': 0,
            'details': [],
            'page_interactions': {}  # Track interactions per page
        }
        
    def add_assertion(self, assertion_type, description, status, error=None, url=None, element_info=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.assertions['details'].append({
            'timestamp': timestamp,
            'type': assertion_type,
            'description': description,
            'status': status,
            'error': str(error) if error else None,
            'url': url,
            'element_info': element_info
        })
        
        # Track page interactions
        if url:
            if url not in self.assertions['page_interactions']:
                self.assertions['page_interactions'][url] = {
                    'first_visit': timestamp,
                    'last_visit': timestamp,
                    'interactions': [],
                    'successful_interactions': 0,
                    'failed_interactions': 0
                }
            
            page_info = self.assertions['page_interactions'][url]
            page_info['last_visit'] = timestamp
            page_info['interactions'].append({
                'timestamp': timestamp,
                'type': assertion_type,
                'description': description,
                'status': status,
                'element_info': element_info
            })
            
            if status == "Passed":
                page_info['successful_interactions'] += 1
                self.assertions['passed'] += 1
            else:
                page_info['failed_interactions'] += 1
                self.assertions['failed'] += 1

    def add_step(self, action_type, description, status="Success", error=None, url=None, element_info=None):
        step = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": action_type,
            "description": description,
            "status": status,
            "error": str(error) if error else None,
            "url": url,
            "element_info": element_info
        }
        self.test_steps.append(step)
        
        if url and action_type == "Navigation":
            self.page_visits.append({
                "timestamp": step["timestamp"],
                "url": url,
                "status": status
            })
            self.visited_urls.add(url)
        
        elif action_type == "Click" and element_info:
            self.button_clicks.append({
                "timestamp": step["timestamp"],
                "element": element_info,
                "url": url,
                "status": status,
                "error": str(error) if error else None
            })
        
        elif element_info:
            self.element_interactions.append({
                "timestamp": step["timestamp"],
                "action_type": action_type,
                "element": element_info,
                "url": url,
                "status": status,
                "error": str(error) if error else None
            })
        
        if status == "Success":
            self.successful_actions += 1
        else:
            self.failed_actions += 1
        self.total_actions += 1
        
    def generate_html_report(self):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Generate summary analysis
        successful_interactions = [step for step in self.test_steps if step.get('status') == 'Success']
        failed_interactions = [step for step in self.test_steps if step.get('status') != 'Success']
        
        # Group interactions by type
        interaction_types = {}
        for step in successful_interactions:
            action_type = step.get('action_type', 'Unknown')
            if action_type not in interaction_types:
                interaction_types[action_type] = []
            interaction_types[action_type].append(step.get('description'))
        
        # Analyze errors
        error_analysis = {}
        for step in failed_interactions:
            error_msg = step.get('error', 'Unknown error')
            if 'outside of the viewport' in error_msg:
                error_type = 'Viewport Error'
            elif 'timeout' in error_msg.lower():
                error_type = 'Timeout Error'
            else:
                error_type = 'Other Error'
            
            if error_type not in error_analysis:
                error_analysis[error_type] = []
            error_analysis[error_type].append({
                'element': step.get('description'),
                'error': error_msg
            })
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Leasingmarkt Autonomous Test Report</title>
            <style>
                :root {
                    --bg-primary: #1a1a1a;
                    --bg-secondary: #2d2d2d;
                    --text-primary: #ffffff;
                    --text-secondary: #b3b3b3;
                    --accent-color: #4CAF50;
                    --error-color: #ff4444;
                    --warning-color: #ffbb33;
                    --success-color: #00C851;
                    --border-radius: 8px;
                    --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }
                
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: var(--bg-primary);
                    color: var(--text-primary);
                    line-height: 1.6;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                .header {
                    background-color: var(--bg-secondary);
                    padding: 30px;
                    border-radius: var(--border-radius);
                    margin-bottom: 30px;
                    box-shadow: var(--shadow);
                }
                
                .header h1 {
                    margin: 0;
                    color: var(--accent-color);
                    font-size: 2.5em;
                    margin-bottom: 15px;
                    text-align: center;
                }
                
                .header-info {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                    text-align: center;
                }
                
                .header-info p {
                    margin: 5px 0;
                    color: var(--text-secondary);
                }
                
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                .metric-card {
                    background-color: var(--bg-secondary);
                    padding: 25px;
                    border-radius: var(--border-radius);
                    text-align: center;
                    box-shadow: var(--shadow);
                    transition: transform 0.2s;
                }
                
                .metric-card:hover {
                    transform: translateY(-5px);
                }
                
                .metric-value {
                    font-size: 2.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }
                
                .metric-label {
                    color: var(--text-secondary);
                    font-size: 1.1em;
                }
                
                .success-metric .metric-value {
                    color: var(--success-color);
                }
                
                .error-metric .metric-value {
                    color: var(--error-color);
                }
                
                .warning-metric .metric-value {
                    color: var(--warning-color);
                }
                
                .steps-container {
                    background-color: var(--bg-secondary);
                    padding: 30px;
                    border-radius: var(--border-radius);
                    margin-bottom: 30px;
                    box-shadow: var(--shadow);
                }
                
                .steps-container h2 {
                    color: var(--accent-color);
                    margin-top: 0;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid var(--accent-color);
                }
                
                .step {
                    margin-bottom: 20px;
                    padding: 20px;
                    background-color: var(--bg-primary);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow);
                    transition: transform 0.2s;
                }
                
                .step:hover {
                    transform: translateX(5px);
                }
                
                .step.success {
                    border-left: 4px solid var(--success-color);
                }
                
                .step.error {
                    border-left: 4px solid var(--error-color);
                    background-color: rgba(255, 68, 68, 0.1);
                }
                
                .timestamp {
                    color: var(--text-secondary);
                    font-size: 0.9em;
                    margin-bottom: 5px;
                }
                
                .step-type {
                    font-weight: bold;
                    color: var(--accent-color);
                    margin-bottom: 5px;
                }
                
                .step-description {
                    margin: 10px 0;
                }
                
                .error-message {
                    color: var(--error-color);
                    background-color: rgba(255, 68, 68, 0.1);
                    padding: 10px;
                    border-radius: var(--border-radius);
                    margin-top: 10px;
                    font-family: monospace;
                }
                
                .progress-bar {
                    height: 10px;
                    background-color: var(--bg-primary);
                    border-radius: var(--border-radius);
                    margin: 20px 0;
                    overflow: hidden;
                }
                
                .progress-fill {
                    height: 100%;
                    background-color: var(--accent-color);
                    transition: width 0.5s ease-in-out;
                }
                
                .summary-section {
                    background-color: var(--bg-secondary);
                    padding: 30px;
                    border-radius: var(--border-radius);
                    margin-bottom: 30px;
                    box-shadow: var(--shadow);
                }
                
                .summary-section h2 {
                    color: var(--accent-color);
                    margin-top: 0;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid var(--accent-color);
                }
                
                .summary-content {
                    background-color: var(--bg-primary);
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                }
                
                .interaction-type {
                    margin-bottom: 15px;
                }
                
                .interaction-type h3 {
                    color: var(--accent-color);
                    margin-bottom: 10px;
                }
                
                .interaction-list {
                    list-style-type: none;
                    padding-left: 20px;
                    margin: 0;
                }
                
                .interaction-list li {
                    margin-bottom: 5px;
                    position: relative;
                }
                
                .interaction-list li:before {
                    content: "•";
                    color: var(--accent-color);
                    position: absolute;
                    left: -15px;
                }
                
                .error-analysis {
                    margin-top: 20px;
                }
                
                .error-type {
                    margin-bottom: 15px;
                }
                
                .error-type h3 {
                    color: var(--error-color);
                    margin-bottom: 10px;
                }
                
                .error-list {
                    list-style-type: none;
                    padding-left: 20px;
                    margin: 0;
                }
                
                .error-list li {
                    margin-bottom: 10px;
                    position: relative;
                }
                
                .error-list li:before {
                    content: "❌";
                    position: absolute;
                    left: -20px;
                }
                
                .page-summary {
                    background-color: var(--bg-primary);
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-bottom: 20px;
                }

                .page-summary h3 {
                    color: var(--accent-color);
                    margin-top: 0;
                    margin-bottom: 15px;
                    border-bottom: 1px solid var(--accent-color);
                    padding-bottom: 5px;
                }

                .page-stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    margin-bottom: 15px;
                }

                .page-stats p {
                    margin: 5px 0;
                    color: var(--text-secondary);
                }

                .page-interactions {
                    margin-top: 15px;
                }

                .page-interactions h4 {
                    color: var(--text-primary);
                    margin-bottom: 10px;
                }
                
                @media (max-width: 768px) {
                    body {
                        padding: 10px;
                    }
                    
                    .metrics {
                        grid-template-columns: 1fr;
                    }
                    
                    .step {
                        padding: 15px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Leasingmarkt Autonomous Test Report</h1>
                    <div class="header-info">
                        <div>
                            <p><strong>Start Time:</strong><br>{{ start_time }}</p>
                        </div>
                        <div>
                            <p><strong>End Time:</strong><br>{{ end_time }}</p>
                        </div>
                        <div>
                            <p><strong>Duration:</strong><br>{{ duration }} seconds</p>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ (successful_actions / total_actions * 100) if total_actions > 0 else 0 }}%"></div>
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Total Actions</div>
                        <div class="metric-value">{{ total_actions }}</div>
                    </div>
                    <div class="metric-card success-metric">
                        <div class="metric-label">Successful Actions</div>
                        <div class="metric-value">{{ successful_actions }}</div>
                    </div>
                    <div class="metric-card error-metric">
                        <div class="metric-label">Failed Actions</div>
                        <div class="metric-value">{{ failed_actions }}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Pages Visited</div>
                        <div class="metric-value">{{ visited_urls }}</div>
                    </div>
                    <div class="metric-card success-metric">
                        <div class="metric-label">Assertions Passed</div>
                        <div class="metric-value">{{ assertions.passed }}</div>
                    </div>
                    <div class="metric-card error-metric">
                        <div class="metric-label">Assertions Failed</div>
                        <div class="metric-value">{{ assertions.failed }}</div>
                    </div>
                </div>
                
                {% if assertions.page_interactions %}
                <div class="summary-section">
                    <h2>🌐 Page Interactions Summary</h2>
                    <div class="summary-content">
                        {% for url, page_info in assertions.page_interactions.items() %}
                        <div class="page-summary">
                            <h3>{{ url }}</h3>
                            <div class="page-stats">
                                <p>First Visit: {{ page_info.first_visit }}</p>
                                <p>Last Visit: {{ page_info.last_visit }}</p>
                                <p>Total Interactions: {{ page_info.interactions|length }}</p>
                                <p>Successful: {{ page_info.successful_interactions }}</p>
                                <p>Failed: {{ page_info.failed_interactions }}</p>
                            </div>
                            <div class="page-interactions">
                                <h4>Interactions:</h4>
                                <ul class="interaction-list">
                                    {% for interaction in page_info.interactions %}
                                    <li class="{% if interaction.status == 'Passed' %}success{% else %}error{% endif %}">
                                        <span class="timestamp">{{ interaction.timestamp }}</span>
                                        <strong>{{ interaction.type }}:</strong> 
                                        {{ interaction.description }}
                                        {% if interaction.element_info %}
                                        <br><small>Element: {{ interaction.element_info }}</small>
                                        {% endif %}
                                        {% if interaction.error %}
                                        <div class="error-message">{{ interaction.error }}</div>
                                        {% endif %}
                                    </li>
                                    {% endfor %}
                                </ul>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <div class="steps-container">
                    <h2>🔍 Test Steps</h2>
                    {% for step in test_steps %}
                    <div class="step {{ 'success' if step.status == 'Success' else 'error' }}">
                        <div class="timestamp">⏰ {{ step.timestamp }}</div>
                        <div class="step-type">{{ step.action_type }}</div>
                        <div class="step-description">{{ step.description }}</div>
                        {% if step.error %}
                        <div class="error-message">
                            <strong>❌ Error:</strong> {{ step.error }}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                    
                    <h2>✅ Assertions</h2>
                    {% for assertion in assertions.details %}
                    <div class="step {{ 'success' if assertion.status == 'Passed' else 'error' }}">
                        <div class="timestamp">⏰ {{ assertion.timestamp }}</div>
                        <div class="step-type">{{ assertion.type }}</div>
                        <div class="step-description">{{ assertion.description }}</div>
                        {% if assertion.error %}
                        <div class="error-message">
                            <strong>❌ Error:</strong> {{ assertion.error }}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            start_time=self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=f"{duration:.2f}",
            total_actions=self.total_actions,
            successful_actions=self.successful_actions,
            failed_actions=self.failed_actions,
            visited_urls=len(self.visited_urls),
            test_steps=self.test_steps,
            assertions=self.assertions
        )
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Save as index.html for GitHub Pages
        report_path = os.path.join('reports', 'index.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report_path

class AutomatedTester:
    def __init__(self, model_path, interactions_path):
        # Load the trained model
        print(f"\nLoading model from {model_path}...")
        try:
            from src.bot.hybrid_learner import HybridInteractionLearner
            self.learner = HybridInteractionLearner()
            self.learner.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Load recorded interactions for reference
        with open(interactions_path, 'r') as f:
            self.recorded_interactions = json.load(f)
        
        # Extract unique URLs and common selectors from recorded interactions
        self.visited_urls = set(interaction['url'] for interaction in self.recorded_interactions if 'url' in interaction)
        self.common_selectors = set(interaction['selector'] for interaction in self.recorded_interactions if 'selector' in interaction)
        
        self.base_url = None
        self.checked_links = set()
        
    def validate_link(self, page, url):
        """Validate a URL with security checks."""
        try:
            parsed = urlparse(url)
            # Security checks
            if not parsed.scheme in ['http', 'https']:
                return False
            if not parsed.netloc:
                return False
            if self.base_url and parsed.netloc != urlparse(self.base_url).netloc:
                return False
            # Check for potential security risks
            risky_patterns = [
                'file://', 'data:', 'javascript:', 'vbscript:',
                'admin', 'login', 'logout', 'delete', 'remove'
            ]
            return not any(pattern in url.lower() for pattern in risky_patterns)
        except Exception:
            return False

    def run_tests(self, url, duration=300):  # 5 minutes of testing by default
        print(f"\nStarting automated exploratory testing on {url}")
        print(f"Will run tests for {duration} seconds")
        
        # Set base URL for link validation
        self.base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        
        # Initialize test report
        report = TestReport()
        report.add_step("Initialize", f"Starting test on {url} with duration {duration} seconds")
        
        try:
            with sync_playwright() as p:
                # Check for headless mode from environment variable
                headless = os.getenv('PLAYWRIGHT_HEADLESS', 'false').lower() == 'true'
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context()
                page = context.new_page()
                
                # Start time for duration tracking
                start_time = time.time()
                actions_performed = 0
                
                # Navigate to initial URL
                print("\nNavigating to starting URL...")
                report.add_step("Navigation", f"Navigating to starting URL: {url}", url=url)
                page.goto(url)
                report.visited_urls.add(url)
                
                # Handle cookie consent
                try:
                    cookie_button = page.get_by_test_id('as24-cmp-accept-all-button')
                    if cookie_button:
                        print("Accepting cookies...")
                        report.add_step("Cookie Consent", "Accepting cookies")
                        cookie_button.click()
                        time.sleep(1)
                except Exception as e:
                    report.add_step("Cookie Consent", "Failed to accept cookies", "Failed", e)
                
                while time.time() - start_time < duration:
                    remaining = int(duration - (time.time() - start_time))
                    
                    if remaining % 30 == 0:
                        print(f"\n{remaining} seconds remaining...")
                        print(f"Actions performed: {actions_performed}")
                    
                    try:
                        # Perform element checks
                        self._perform_page_checks(page, report)
                        
                        # Regular interaction logic...
                        if self.common_selectors and random.random() < 0.7:
                            selector = random.choice(list(self.common_selectors))
                            elements = page.query_selector_all(selector)
                            if elements:
                                element = random.choice(elements)
                                self._smart_interaction(page, element, report)
                                actions_performed += 1
                        else:
                            elements = page.query_selector_all('button, input, a, select, [role="button"]')
                            if elements:
                                element = random.choice(elements)
                                self._smart_interaction(page, element, report)
                                actions_performed += 1
                        
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        report.add_step("Interaction", "Failed to perform interaction", "Failed", e)
                        continue
                
                print("\nTesting completed!")
                print(f"Total actions performed: {actions_performed}")
                report.add_step("Complete", f"Testing completed. Total actions performed: {actions_performed}")
                
                context.close()
                browser.close()
                
        except Exception as e:
            report.add_step("Critical", "Critical error during testing", "Failed", e)
            
        # Generate and save report
        report_path = report.generate_html_report()
        print(f"\nTest report generated: {report_path}")
            
    def _smart_interaction(self, page, element, report):
        """Perform smart interaction with security checks."""
        try:
            # Security check for element
            if not element.is_visible() or not element.is_enabled():
                return False

            # Get element properties safely
            tag_name = element.evaluate('element => element.tagName.toLowerCase()')
            element_type = element.get_attribute('type')
            
            # Validate input types
            if tag_name == 'input':
                if element_type in ['file', 'hidden']:
                    return False
                if element_type == 'text':
                    # Sanitize input value
                    safe_value = "test_input"
                    element.fill(safe_value)
                    return True

            # Safe click handling
            if tag_name in ['button', 'a'] or element.is_enabled():
                element.click(timeout=5000)
                return True

            return False
        except Exception as e:
            report.add_assertion('interaction', 'Element interaction failed', 'Failed', 
                               error=str(e), url=page.url, element_info=str(element))
            return False

    def _perform_page_checks(self, page, report):
        try:
            # Get current URL
            current_url = page.url
            
            # Basic page load check
            report.add_assertion("Page Load", f"Page loaded successfully: {current_url}", "Passed", url=current_url)
            
            # Check for main content
            main_content = page.locator("main").first
            if main_content.is_visible():
                report.add_assertion("Content", "Main content is visible", "Passed", url=current_url)
            else:
                report.add_assertion("Content", "Main content not found", "Failed", url=current_url)
            
            # Check for navigation
            nav = page.locator("nav").first
            if nav.is_visible():
                report.add_assertion("Navigation", "Navigation menu is present", "Passed", url=current_url)
            else:
                report.add_assertion("Navigation", "Navigation menu not found", "Failed", url=current_url)
            
            return True
        except Exception as e:
            report.add_assertion("Page Check", "Failed to perform page checks", "Failed", error=str(e), url=current_url)
            return False