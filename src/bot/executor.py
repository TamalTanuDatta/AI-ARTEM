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
                    --success-color: #00C851;
                    --border-radius: 8px;
                }
                
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: var(--bg-primary);
                    color: var(--text-primary);
                    line-height: 1.6;
                }
                
                .header {
                    background-color: var(--bg-secondary);
                    padding: 30px;
                    border-radius: var(--border-radius);
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }
                
                .header h1 {
                    margin: 0;
                    color: var(--accent-color);
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                
                .header p {
                    margin: 5px 0;
                    color: var(--text-secondary);
                }
                
                .summary {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                .summary-box {
                    background-color: var(--bg-secondary);
                    padding: 20px;
                    border-radius: var(--border-radius);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    transition: transform 0.2s;
                }
                
                .summary-box:hover {
                    transform: translateY(-2px);
                }
                
                .summary-box h3 {
                    margin: 0;
                    color: var(--accent-color);
                    font-size: 1.2em;
                    margin-bottom: 10px;
                }
                
                .summary-box p {
                    margin: 0;
                    font-size: 2em;
                    font-weight: bold;
                }
                
                .steps, .assertions, .visited-urls {
                    background-color: var(--bg-secondary);
                    padding: 20px;
                    border-radius: var(--border-radius);
                    margin-bottom: 30px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                }
                
                h2 {
                    color: var(--accent-color);
                    margin-top: 0;
                    margin-bottom: 20px;
                    border-bottom: 2px solid var(--accent-color);
                    padding-bottom: 10px;
                }
                
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    background-color: var(--bg-primary);
                    border-radius: var(--border-radius);
                    overflow: hidden;
                }
                
                th, td {
                    padding: 15px;
                    text-align: left;
                    border-bottom: 1px solid var(--bg-secondary);
                }
                
                th {
                    background-color: var(--bg-secondary);
                    color: var(--accent-color);
                    font-weight: 600;
                }
                
                tr:hover {
                    background-color: rgba(76, 175, 80, 0.1);
                }
                
                .success {
                    color: var(--success-color);
                }
                
                .error {
                    color: var(--error-color);
                }
                
                .timestamp {
                    color: var(--text-secondary);
                    font-size: 0.9em;
                }
                
                .visited-urls ul {
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                }
                
                .visited-urls li {
                    padding: 10px;
                    border-bottom: 1px solid var(--bg-primary);
                }
                
                .visited-urls li:last-child {
                    border-bottom: none;
                }
                
                .visited-urls li:hover {
                    background-color: rgba(76, 175, 80, 0.1);
                }
                
                @media (max-width: 768px) {
                    .summary {
                        grid-template-columns: 1fr;
                    }
                    
                    body {
                        padding: 10px;
                    }
                    
                    .header {
                        padding: 20px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Leasingmarkt Autonomous Test Report</h1>
                <p>Test executed from {{ start_time }} to {{ end_time }}</p>
                <p>Total Duration: {{ duration }} seconds</p>
            </div>
            
            <div class="summary">
                <div class="summary-box">
                    <h3>Total Actions</h3>
                    <p>{{ total_actions }}</p>
                </div>
                <div class="summary-box">
                    <h3>Successful Actions</h3>
                    <p class="success">{{ successful_actions }}</p>
                </div>
                <div class="summary-box">
                    <h3>Failed Actions</h3>
                    <p class="error">{{ failed_actions }}</p>
                </div>
                <div class="summary-box">
                    <h3>Total Assertions</h3>
                    <p>{{ assertions.passed + assertions.failed }}</p>
                </div>
                <div class="summary-box">
                    <h3>Passed Assertions</h3>
                    <p class="success">{{ assertions.passed }}</p>
                </div>
                <div class="summary-box">
                    <h3>Failed Assertions</h3>
                    <p class="error">{{ assertions.failed }}</p>
                </div>
            </div>
            
            <div class="steps">
                <h2>Test Steps</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Action Type</th>
                        <th>Description</th>
                        <th>Status</th>
                        <th>Error</th>
                    </tr>
                    {% for step in steps %}
                    <tr>
                        <td class="timestamp">{{ step.timestamp }}</td>
                        <td>{{ step.action_type }}</td>
                        <td>{{ step.description }}</td>
                        <td class="{{ 'success' if step.status == 'Success' else 'error' }}">{{ step.status }}</td>
                        <td class="error">{{ step.error if step.error else '' }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="assertions">
                <h2>Assertions</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Description</th>
                        <th>Status</th>
                        <th>Error</th>
                    </tr>
                    {% for assertion in assertions.details %}
                    <tr>
                        <td class="timestamp">{{ assertion.timestamp }}</td>
                        <td>{{ assertion.type }}</td>
                        <td>{{ assertion.description }}</td>
                        <td class="{{ 'success' if assertion.status == 'Passed' else 'error' }}">{{ assertion.status }}</td>
                        <td class="error">{{ assertion.error if assertion.error else '' }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="visited-urls">
                <h2>Visited URLs</h2>
                <ul>
                    {% for url in visited_urls %}
                    <li>{{ url }}</li>
                    {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        report_html = template.render(
            start_time=self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=round(duration, 2),
            total_actions=self.total_actions,
            successful_actions=self.successful_actions,
            failed_actions=self.failed_actions,
            steps=self.test_steps,
            assertions=self.assertions,
            visited_urls=sorted(list(self.visited_urls))
        )
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Delete old reports
        old_reports = glob.glob("reports/leasingmarkt_test_report_*.html")
        for report in old_reports:
            try:
                os.remove(report)
                print(f"Deleted old report: {report}")
            except Exception as e:
                print(f"Warning: Could not delete old report {report}: {str(e)}")
        
        # Generate unique filename with timestamp
        filename = f"reports/leasingmarkt_test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, "w") as f:
            f.write(report_html)
            
        return filename

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
        
    async def validate_link(self, page, url):
        try:
            if url in self.checked_links:
                return True
                
            # Only check links from the same domain
            if self.base_url and not url.startswith(self.base_url):
                return True
                
            response = await page.request.head(url)
            is_valid = response.status < 400
            self.checked_links.add(url)
            return is_valid
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
                browser = p.chromium.launch(headless=False)
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
                        visible_inputs = [input_field for input_field in inputs if input_field.is_visible()]
                        if visible_inputs:
                            report.add_assertion("Form", f"Form and its inputs are accessible", "Passed")
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