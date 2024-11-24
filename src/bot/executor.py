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
        self.assertions = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
    def add_assertion(self, assertion_type, description, status, error=None):
        self.assertions['details'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': assertion_type,
            'description': description,
            'status': status,
            'error': str(error) if error else None
        })
        if status == "Passed":
            self.assertions['passed'] += 1
        else:
            self.assertions['failed'] += 1

    def add_step(self, action_type, description, status="Success", error=None):
        step = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": action_type,
            "description": description,
            "status": status,
            "error": str(error) if error else None
        }
        self.test_steps.append(step)
        
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
                
                <div class="summary-section">
                    <h2>📊 Test Summary Analysis</h2>
                    <div class="summary-content">
                        <p>The bot ran for {{ duration }} seconds and performed {{ total_actions }} actions.</p>
                        
                        <div class="interaction-type">
                            <h3>🎯 Successful Interactions</h3>
                            <ul class="interaction-list">
                            {% for type, interactions in interaction_types.items() %}
                                <li>
                                    <strong>{{ type }}:</strong>
                                    <ul class="interaction-list">
                                    {% for interaction in interactions %}
                                        <li>{{ interaction }}</li>
                                    {% endfor %}
                                    </ul>
                                </li>
                            {% endfor %}
                            </ul>
                        </div>
                        
                        {% if error_analysis %}
                        <div class="error-analysis">
                            <h3>❌ Error Analysis</h3>
                            {% for error_type, errors in error_analysis.items() %}
                            <div class="error-type">
                                <h4>{{ error_type }} ({{ errors|length }})</h4>
                                <ul class="error-list">
                                {% for error in errors %}
                                    <li>
                                        <strong>{{ error.element }}</strong>
                                        <div class="error-message">{{ error.error }}</div>
                                    </li>
                                {% endfor %}
                                </ul>
                            </div>
                            {% endfor %}
                        </div>
                        {% endif %}
                    </div>
                </div>
                
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
            assertions=self.assertions,
            interaction_types=interaction_types,
            error_analysis=error_analysis
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
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['encoder']
        
        # Load recorded interactions for reference
        with open(interactions_path, 'r') as f:
            self.recorded_interactions = json.load(f)
        
        # Extract unique URLs and common selectors from recorded interactions
        self.visited_urls = set(interaction['url'] for interaction in self.recorded_interactions if 'url' in interaction)
        self.common_selectors = set(interaction['selector'] for interaction in self.recorded_interactions if 'selector' in interaction)
        
        self.base_url = None
        self.checked_links = set()
        
    def validate_link(self, page, url):
        try:
            if url in self.checked_links:
                return True
                
            # Only check links from the same domain
            if self.base_url and not url.startswith(self.base_url):
                return True
                
            try:
                response = page.request.head(url)
                is_valid = response.status < 400
                self.checked_links.add(url)
                return is_valid
            except Exception:
                return False
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
                report.add_step("Navigation", f"Navigating to starting URL: {url}")
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
            
    def _perform_page_checks(self, page, report):
        try:
            # 1. Basic Page Structure Check
            body = page.locator("body")
            if body:
                report.add_assertion("Page Structure", "Body element is present", "Passed")
            
            # 2. Navigation Elements
            nav_elements = page.query_selector_all("nav, header, .navigation")
            if nav_elements:
                report.add_assertion("Navigation", "Navigation elements are present", "Passed")
            
            # 3. Interactive Elements
            buttons = page.query_selector_all("button, [role='button'], .btn")
            links = page.query_selector_all("a[href]")
            
            # Check buttons
            for button in buttons[:5]:  # Check first 5 buttons
                try:
                    if button.is_visible() and button.is_enabled():
                        report.add_assertion("Button", f"Button '{button.text_content()[:30]}...' is accessible", "Passed")
                except Exception as e:
                    report.add_assertion("Button", f"Button validation failed", "Failed", e)
            
            # 4. Link Validation
            for link in links[:5]:  # Check first 5 links
                try:
                    href = link.get_attribute("href")
                    if href and not href.startswith(("#", "javascript:", "mailto:")):
                        absolute_url = urljoin(self.base_url, href)
                        if self.validate_link(page, absolute_url):
                            report.add_assertion("Link", f"Valid link: {absolute_url[:50]}...", "Passed")
                        else:
                            report.add_assertion("Link", f"Invalid link: {absolute_url[:50]}...", "Failed")
                except Exception as e:
                    report.add_assertion("Link", "Link validation failed", "Failed", e)
            
            # 5. Content Check
            main_content = page.query_selector("main, #content, .main-content, article")
            if main_content and main_content.is_visible():
                report.add_assertion("Content", "Main content area is present", "Passed")
            
            # 6. Form Check
            forms = page.query_selector_all("form")
            for form in forms[:2]:  # Check first 2 forms
                try:
                    if form.is_visible():
                        inputs = form.query_selector_all("input:not([type='hidden'])")
                        if len(inputs) > 0:
                            report.add_assertion("Form", f"Form with {len(inputs)} visible inputs is accessible", "Passed")
                        else:
                            report.add_assertion("Form", "Form has no visible inputs", "Failed")
                except Exception as e:
                    report.add_assertion("Form", "Form validation failed", "Failed", e)
                    
        except Exception as e:
            report.add_assertion("Page", "Failed to perform page checks", "Failed", e)
            
    def _smart_interaction(self, page, element, report):
        try:
            # Extract element information
            tag_name = element.evaluate('el => el.tagName').lower()
            element_type = element.evaluate('el => el.type')
            
            # Check if element is interactive
            if not (element.is_visible() and element.is_enabled()):
                report.add_assertion("Element", f"Element {tag_name} is not interactive", "Failed")
                return
            
            report.add_assertion("Element", f"Element {tag_name} is interactive", "Passed")
            
            # Get element text and attributes for logging
            element_text = element.evaluate('el => el.textContent')
            element_html = element.evaluate('el => el.outerHTML')
            
            print(f"\nInteracting with {tag_name} element: {element_text[:50]}...")
            
            if tag_name == 'input':
                if element_type in ['text', 'email', 'search']:
                    test_inputs = ['Test', 'test@example.com', 'Search term']
                    input_value = random.choice(test_inputs)
                    element.fill(input_value)
                    report.add_assertion("Input", f"Input value set: {input_value}", "Passed")
                    
                elif element_type == 'checkbox':
                    element.check()
                    report.add_assertion("Checkbox", "Checkbox checked", "Passed")
                    
            elif tag_name == 'select':
                options = element.evaluate('el => Array.from(el.options).map(o => o.value)')
                if options:
                    value = random.choice(options)
                    element.select_option(value=value)
                    report.add_assertion("Select", f"Option selected: {value}", "Passed")
                    
            elif tag_name == 'button' or tag_name == 'a':
                # For links, validate href before clicking
                if tag_name == 'a':
                    href = element.get_attribute("href")
                    if href and not href.startswith(("#", "javascript:", "mailto:")):
                        absolute_url = urljoin(self.base_url, href)
                        if self.validate_link(page, absolute_url):
                            report.add_assertion("Link", f"Valid link target: {absolute_url[:50]}...", "Passed")
                        else:
                            report.add_assertion("Link", f"Invalid link target: {absolute_url[:50]}...", "Failed")
                            return
                
                element.click()
                report.add_assertion("Click", f"Successfully clicked {tag_name}", "Passed")
                
            else:
                if 'button' in element_html or 'click' in element_html:
                    element.click()
                    report.add_assertion("Click", "Successfully clicked interactive element", "Passed")
                    
        except Exception as e:
            report.add_assertion("Interaction", f"Failed to interact with {tag_name} element", "Failed", e)
            print(f"Error during smart interaction: {str(e)}")