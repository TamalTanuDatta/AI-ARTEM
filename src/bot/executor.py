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

         

    def add_assertion(self, assertion_type, description, status='Passed', url=None, error=None): 

        """Add an assertion to the test report.""" 

        self.assertions['details'].append({ 

            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 

            'type': assertion_type, 

            'description': description, 

            'status': status, 

            'url': url, 

            'error': error 

        }) 

         

        if status == 'Passed': 

            self.assertions['passed'] += 1 

        else: 

            self.assertions['failed'] += 1 

             

        # Track page interactions 

        if url: 

            if url not in self.assertions['page_interactions']: 

                self.assertions['page_interactions'][url] = { 

                    'first_visit': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 

                    'last_visit': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 

                    'interactions': [], 

                    'successful_interactions': 0, 

                    'failed_interactions': 0 

                } 

             

            page_info = self.assertions['page_interactions'][url] 

            page_info['last_visit'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

            page_info['interactions'].append({ 

                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 

                'type': assertion_type, 

                'description': description, 

                'status': status, 

                'error': error 

            }) 

             

            if status == "Passed": 

                page_info['successful_interactions'] += 1 

                self.assertions['passed'] += 1 

            else: 

                page_info['failed_interactions'] += 1 

                self.assertions['failed'] += 1 

 

    def add_step(self, action_type, description, status="Success", error=None, url=None, element_info=None): 

        """Add a test step to the report.""" 

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

        """Generate an HTML report of the test execution.""" 

         

        template = Template(""" 

        <!DOCTYPE html> 

        <html> 

        <head> 

            <title>AI-ARTEM (Autonomous Testing Machine) Test Report</title> 

            <style> 

                body { 

                    font-family: Arial, sans-serif; 

                    line-height: 1.6; 

                    margin: 0; 

                    padding: 20px; 

                    background: #1e1e1e; 

                    color: #ffffff; 

                } 

                .container { 

                    max-width: 1200px; 

                    margin: 0 auto; 

                } 

                h1, h2, h3 { 

                    color: #00ff9d; 

                } 

 

                /* Title Animation Styles */ 

                .animated-title { 

                    font-size: 2.5em; 

                    text-align: center; 

                    margin: 20px 0; 

                    background: linear-gradient(120deg, #00ff9d, #00a8ff, #00ff9d); 

                    background-size: 200% auto; 

                    color: transparent; 

                    -webkit-background-clip: text; 

                    background-clip: text; 

                    animation: shine 3s linear infinite, float 3s ease-in-out infinite; 

                    text-shadow: 0 0 10px rgba(0, 255, 157, 0.3); 

                } 

 

                @keyframes shine { 

                    to { 

                        background-position: 200% center; 

                    } 

                } 

 

                @keyframes float { 

                    0%, 100% { 

                        transform: translateY(0); 

                    } 

                    50% { 

                        transform: translateY(-10px); 

                    } 

                } 

 

                .title-container { 

                    position: relative; 

                    margin-bottom: 40px; 

                } 

 

                .title-container::after { 

                    content: ''; 

                    position: absolute; 

                    bottom: -10px; 

                    left: 50%; 

                    transform: translateX(-50%); 

                    width: 0; 

                    height: 3px; 

                    background: linear-gradient(90deg, transparent, #00ff9d, transparent); 

                    animation: expand 2s ease-out forwards; 

                } 

 

                @keyframes expand { 

                    to { 

                        width: 80%; 

                    } 

                } 

 

                .summary-box { 

                    background: #2d2d2d; 

                    border-radius: 8px; 

                    padding: 20px; 

                    margin: 20px 0; 

                } 

                .test-duration { 

                    display: flex; 

                    justify-content: space-between; 

                    background: #3d3d3d; 

                    padding: 15px; 

                    border-radius: 6px; 

                    margin: 20px 0; 

                } 

                .time-block { 

                    text-align: center; 

                } 

                .time-label { 

                    color: #888; 

                    font-size: 0.9em; 

                } 

                .time-value { 

                    font-size: 1.2em; 

                    color: #00ff9d; 

                } 

                .stat-grid { 

                    display: grid; 

                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 

                    gap: 20px; 

                    margin: 20px 0; 

                } 

                .stat-card { 

                    background: #3d3d3d; 

                    padding: 15px; 

                    border-radius: 6px; 

                    text-align: center; 

                } 

                .stat-number { 

                    font-size: 24px; 

                    font-weight: bold; 

                    margin: 10px 0; 

                } 

                .success { 

                    color: #4caf50; 

                } 

                .error { 

                    color: #f44336; 

                } 

                .warning { 

                    color: #ff9800; 

                } 

                .timeline { 

                    margin: 20px 0; 

                } 

                .timeline-item { 

                    background: #2d2d2d; 

                    padding: 15px; 

                    margin: 10px 0; 

                    border-radius: 6px; 

                } 

                .passed { 

                    border-left: 4px solid #4caf50; 

                } 

                .failed { 

                    border-left: 4px solid #f44336; 

                } 

                .timestamp { 

                    color: #888; 

                    font-size: 0.9em; 

                } 

                .details { 

                    margin-top: 10px; 

                    color: #ccc; 

                } 

                .failed-list { 

                    background: #3d2d2d; 

                    padding: 15px; 

                    border-radius: 6px; 

                    margin: 10px 0; 

                } 

                .collapsible { 

                    cursor: pointer; 

                    padding: 18px 25px; 

                    width: 100%; 

                    border: none; 

                    text-align: left; 

                    outline: none; 

                    background: linear-gradient(145deg, #3d3d3d, #2d2d2d); 

                    color: white; 

                    border-radius: 10px; 

                    margin-top: 15px; 

                    font-size: 16px; 

                    font-weight: 500; 

                    display: flex; 

                    justify-content: space-between; 

                    align-items: center; 

                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 

                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); 

                    position: relative; 

                    overflow: hidden; 

                } 

 

                .collapsible:hover { 

                    background: linear-gradient(145deg, #4d4d4d, #3d3d3d); 

                    transform: translateY(-2px); 

                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); 

                } 

 

                .collapsible:active { 

                    transform: translateY(0); 

                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); 

                } 

 

                .collapsible:after { 

                    content: '\\002B'; 

                    color: #00ff9d; 

                    font-weight: bold; 

                    margin-left: 15px; 

                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 

                    font-size: 20px; 

                } 

 

                .active:after { 

                    content: "\\2212"; 

                    transform: rotate(180deg); 

                    color: #00a8ff; 

                } 

 

                .content { 

                    max-height: 0; 

                    overflow: hidden; 

                    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1); 

                    background-color: #2d2d2d; 

                    margin: 0 5px 10px 5px; 

                    border-bottom-left-radius: 10px; 

                    border-bottom-right-radius: 10px; 

                    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.1); 

                } 

 

                .content .timeline { 

                    padding: 25px; 

                    margin: 0; 

                    opacity: 0; 

                    transform: translateY(-10px); 

                    transition: all 0.3s ease-out; 

                } 

 

                .content.show .timeline { 

                    opacity: 1; 

                    transform: translateY(0); 

                } 

 

                .timeline-item { 

                    background: linear-gradient(145deg, #363636, #2d2d2d); 

                    padding: 18px 25px; 

                    margin: 12px 0; 

                    border-radius: 10px; 

                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); 

                    transition: all 0.3s ease; 

                } 

 

                .timeline-item:hover { 

                    transform: translateX(5px); 

                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); 

                } 

            </style> 

            <script> 

                function toggleCollapsible(element) { 

                    element.classList.toggle("active"); 

                    var content = element.nextElementSibling; 

                    if (content.style.maxHeight) { 

                        content.style.maxHeight = null; 

                        content.classList.remove("show"); 

                        setTimeout(() => { 

                            content.querySelector(".timeline").style.opacity = "0"; 

                            content.querySelector(".timeline").style.transform = "translateY(-10px)"; 

                        }, 200); 

                    } else { 

                        content.style.maxHeight = content.scrollHeight + "px"; 

                        content.classList.add("show"); 

                        setTimeout(() => { 

                            content.querySelector(".timeline").style.opacity = "1"; 

                            content.querySelector(".timeline").style.transform = "translateY(0)"; 

                        }, 200); 

                    } 

                } 

            </script> 

        </head> 

        <body> 

            <div class="container"> 

                <div class="title-container"> 

                    <h1 class="animated-title">AI-ARTEM (Autonomous Testing Machine) Test Report</h1> 

                </div> 

                 

                <div class="summary-box"> 

                    <h2>Test Timeline</h2> 

                    <div class="test-duration"> 

                        <div class="time-block"> 

                            <div class="time-label">Start Time</div> 

                            <div class="time-value">{{ start_time.strftime('%Y-%m-%d %H:%M:%S') }}</div> 

                        </div> 

                        <div class="time-block"> 

                            <div class="time-label">Duration</div> 

                            <div class="time-value">{{ "%.2f"|format((end_time - start_time).total_seconds()) }}s</div> 

                        </div> 

                        <div class="time-block"> 

                            <div class="time-label">End Time</div> 

                            <div class="time-value">{{ end_time.strftime('%Y-%m-%d %H:%M:%S') }}</div> 

                        </div> 

                    </div> 

                </div> 

 

                <div class="summary-box"> 

                    <h2>Test Summary</h2> 

                    <div class="stat-grid"> 

                        <div class="stat-card"> 

                            <h3>Test Steps</h3> 

                            <div class="stat-number">{{ total_actions }}</div> 

                        </div> 

                        <div class="stat-card"> 

                            <h3>Total Assertions</h3> 

                            <div class="stat-number">{{ assertions['passed'] + assertions['failed'] }}</div> 

                        </div> 

                        <div class="stat-card"> 

                            <h3>Success Rate</h3> 

                            <div class="stat-number {% if assertions['passed'] / (assertions['passed'] + assertions['failed']) > 0.8 %}success{% else %}warning{% endif %}"> 

                                {{ "%.1f"|format(assertions['passed'] / (assertions['passed'] + assertions['failed']) * 100) }}% 

                            </div> 

                        </div> 

                    </div> 

                </div> 

 

                <div class="summary-box"> 

                    <h2>Test Results</h2> 

                     

                    <button class="collapsible" onclick="toggleCollapsible(this)"> 

                        Passed Assertions ({{ assertions['passed'] }}) 

                        <span class="stat-number success" style="float: right; font-size: 16px;"> 

                            {{ "%.1f"|format(assertions['passed'] / (assertions['passed'] + assertions['failed']) * 100) }}% 

                        </span> 

                    </button> 

                    <div class="content"> 

                        <div class="timeline"> 

                            {% for assertion in assertions['details'] %} 

                                {% if assertion['status'] == 'Passed' %} 

                                <div class="timeline-item passed"> 

                                    <strong>{{ assertion['type'] }}</strong> 

                                    <div class="timestamp">{{ assertion['timestamp'] }}</div> 

                                    <div class="details"> 

                                        {{ assertion['description'] }} 

                                        {% if assertion['url'] %} 

                                        <div>URL: {{ assertion['url'] }}</div> 

                                        {% endif %} 

                                    </div> 

                                </div> 

                                {% endif %} 

                            {% endfor %} 

                        </div> 

                    </div> 

 

                    <button class="collapsible" onclick="toggleCollapsible(this)"> 

                        Failed Assertions ({{ assertions['failed'] }}) 

                        <span class="stat-number error" style="float: right; font-size: 16px;"> 

                            {{ "%.1f"|format(assertions['failed'] / (assertions['passed'] + assertions['failed']) * 100) }}% 

                        </span> 

                    </button> 

                    <div class="content"> 

                        <div class="timeline"> 

                            {% for assertion in assertions['details'] %} 

                                {% if assertion['status'] == 'Failed' %} 

                                <div class="timeline-item failed"> 

                                    <strong>{{ assertion['type'] }}</strong> 

                                    <div class="timestamp">{{ assertion['timestamp'] }}</div> 

                                    <div class="details"> 

                                        {{ assertion['description'] }} 

                                        {% if assertion['error'] %} 

                                        <div class="error">Error: {{ assertion['error'] }}</div> 

                                        {% endif %} 

                                        {% if assertion['url'] %} 

                                        <div>URL: {{ assertion['url'] }}</div> 

                                        {% endif %} 

                                    </div> 

                                </div> 

                                {% endif %} 

                            {% endfor %} 

                        </div> 

                    </div> 

                </div> 

 

                <div class="summary-box"> 

                    <h2>Complete Timeline</h2> 

                    <div class="timeline"> 

                        {% for assertion in assertions['details'] %} 

                        <div class="timeline-item {{ assertion['status'].lower() }}"> 

                            <strong>{{ assertion['type'] }}</strong> 

                            <div class="timestamp">{{ assertion['timestamp'] }}</div> 

                            <div class="details"> 

                                {{ assertion['description'] }} 

                                {% if assertion['error'] %} 

                                <div class="error">Error: {{ assertion['error'] }}</div> 

                                {% endif %} 

                                {% if assertion['url'] %} 

                                <div>URL: {{ assertion['url'] }}</div> 

                                {% endif %} 

                            </div> 

                        </div> 

                        {% endfor %} 

                    </div> 

                </div> 

            </div> 

        </body> 

        </html> 

        """) 

         

        # Create reports directory if it doesn't exist 

        os.makedirs('reports', exist_ok=True) 

         

        # Generate the report 

        report_path = 'reports/index.html' 

        html_content = template.render( 

            assertions=self.assertions, 

            total_actions=self.total_actions, 

            successful_actions=self.successful_actions, 

            failed_actions=self.failed_actions, 

            start_time=self.start_time, 

            end_time=datetime.now() 

        ) 

         

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

        try: 

            # Get the current URL before interaction 

            current_url = page.url 

             

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

 

class InteractionExecutor: 

    def __init__(self): 

        self.report = TestReport() 

         

    def handle_cookie_acceptance(self, page, url): 

        """Handle cookie acceptance if the banner is present.""" 

        try: 

            # Wait for cookie banner with timeout 

            cookie_button = page.get_by_test_id('as24-cmp-accept-all-button') 

            if cookie_button.is_visible(timeout=5000): 

                cookie_button.click() 

                page.wait_for_load_state('networkidle') 

                self.report.add_assertion( 

                    'Cookie Acceptance', 

                    'Successfully accepted cookies', 

                    status='Passed', 

                    url=url 

                ) 

            else: 

                self.report.add_assertion( 

                    'Cookie Banner', 

                    'Cookie banner not present or already accepted', 

                    status='Passed', 

                    url=url 

                ) 

        except Exception as e: 

            self.report.add_assertion( 

                'Cookie Acceptance', 

                'Failed to handle cookie acceptance', 

                status='Failed', 

                url=url, 

                error=str(e) 

            ) 

             

    def execute_interactions(self, learner, duration=300): 

        """ 

        Execute learned interactions using the trained model. 

         

        Args: 

            learner: Trained HybridInteractionLearner instance 

            duration (int): Maximum duration to run tests in seconds 

        """ 

        print(f"\nExecuting automated tests for {duration} seconds...") 

         

        try: 

            with sync_playwright() as p: 

                # Use headless mode if in CI environment 

                is_ci = os.getenv('GITHUB_ACTIONS') == 'true' 

                browser = p.chromium.launch(headless=is_ci) 

                context = browser.new_context() 

                page = context.new_page() 

                 

                # Start from the default URL 

                url = "https://www.leasingmarkt.de" 

                print(f"Opening {url}") 

                 

                # Assert initial navigation 

                try: 

                    response = page.goto(url, wait_until="networkidle") 

                    if response.ok: 

                        self.report.add_assertion( 

                            'Initial Navigation', 

                            'Successfully loaded homepage', 

                            status='Passed', 

                            url=url 

                        ) 

                    else: 

                        self.report.add_assertion( 

                            'Initial Navigation', 

                            f'Failed to load homepage: Status {response.status}', 

                            status='Failed', 

                            url=url, 

                            error=f"HTTP {response.status}" 

                        ) 

                except Exception as e: 

                    self.report.add_assertion( 

                        'Initial Navigation', 

                        'Failed to load homepage', 

                        status='Failed', 

                        url=url, 

                        error=str(e) 

                    ) 

                    return 

                 

                # Verify page title 

                try: 

                    title = page.title() 

                    if "Leasingmarkt" in title: 

                        self.report.add_assertion( 

                            'Page Verification', 

                            f'Correct page title found: {title}', 

                            status='Passed', 

                            url=url 

                        ) 

                    else: 

                        self.report.add_assertion( 

                            'Page Verification', 

                            f'Unexpected page title: {title}', 

                            status='Failed', 

                            url=url 

                        ) 

                except Exception as e: 

                    self.report.add_assertion( 

                        'Page Verification', 

                        'Failed to verify page title', 

                        status='Failed', 

                        url=url, 

                        error=str(e) 

                    ) 

                 

                # Handle cookie acceptance 

                self.handle_cookie_acceptance(page, url) 

                 

                start_time = time.time() 

                while time.time() - start_time < duration: 

                    try: 

                        # Verify page is responsive 

                        try: 

                            page.wait_for_selector('body', timeout=5000) 

                            self.report.add_assertion( 

                                'Page Responsiveness', 

                                'Page is responsive', 

                                status='Passed', 

                                url=page.url 

                            ) 

                        except Exception as e: 

                            self.report.add_assertion( 

                                'Page Responsiveness', 

                                'Page is not responsive', 

                                status='Failed', 

                                url=page.url, 

                                error=str(e) 

                            ) 

                            continue 

                         

                        # Check for error messages 

                        error_messages = page.query_selector_all('.error, .alert-error, [role="alert"]') 

                        if error_messages: 

                            for error in error_messages: 

                                error_text = error.inner_text() 

                                self.report.add_assertion( 

                                    'Error Detection', 

                                    f'Found error message: {error_text}', 

                                    status='Failed', 

                                    url=page.url, 

                                    error=error_text 

                                ) 

                         

                        # Get all clickable elements 

                        elements = page.query_selector_all('a, button, input[type="submit"], [role="button"], .clickable') 

                         

                        # Assert elements found 

                        if elements: 

                            visible_elements = [e for e in elements if e.is_visible()] 

                            self.report.add_assertion( 

                                'Element Detection', 

                                f'Found {len(visible_elements)} visible interactive elements out of {len(elements)} total', 

                                status='Passed', 

                                url=page.url 

                            ) 

                             

                            if not visible_elements: 

                                self.report.add_assertion( 

                                    'Element Visibility', 

                                    'No visible interactive elements found', 

                                    status='Failed', 

                                    url=page.url 

                                ) 

                                continue 

                                 

                            elements = visible_elements 

                        else: 

                            self.report.add_assertion( 

                                'Element Detection', 

                                'No interactive elements found on page', 

                                status='Failed', 

                                url=page.url 

                            ) 

                            continue 

                             

                        # Randomly select an element to interact with 

                        element = random.choice(elements) 

                         

                        # Get element properties 

                        tag_name = element.evaluate('el => el.tagName').lower() 

                        element_text = element.evaluate('el => el.textContent || el.value || ""').strip() 

                        href = element.get_attribute('href') if tag_name == 'a' else None 

                         

                        # Verify element is interactive 

                        try: 

                            is_enabled = element.is_enabled() 

                            is_visible = element.is_visible() 

                             

                            if not (is_enabled and is_visible): 

                                self.report.add_assertion( 

                                    'Element State', 

                                    f'{tag_name} element is not interactive (enabled: {is_enabled}, visible: {is_visible})', 

                                    status='Failed', 

                                    url=page.url 

                                ) 

                                continue 

                                 

                            self.report.add_assertion( 

                                'Element State', 

                                f'{tag_name} element is interactive: {element_text[:50]}', 

                                status='Passed', 

                                url=page.url 

                            ) 

                        except Exception as e: 

                            self.report.add_assertion( 

                                'Element State', 

                                f'Failed to verify {tag_name} element state', 

                                status='Failed', 

                                url=page.url, 

                                error=str(e) 

                            ) 

                            continue 

                         

                        # Record the interaction 

                        self.report.add_step( 

                            action_type="Click", 

                            description=f"Clicking {tag_name} element: {element_text[:50]}", 

                            url=page.url, 

                            element_info=str(element) 

                        ) 

                         

                        # Perform the interaction 

                        if href and href.startswith(('http', '/')): 

                            try: 

                                response = page.goto(urljoin(url, href), wait_until="networkidle") 

                                if response.ok: 

                                    self.report.add_assertion( 

                                        'Navigation', 

                                        f'Successfully navigated to: {page.url}', 

                                        status='Passed', 

                                        url=page.url 

                                    ) 

                                else: 

                                    self.report.add_assertion( 

                                        'Navigation', 

                                        f'Navigation failed with status: {response.status}', 

                                        status='Failed', 

                                        url=page.url, 

                                        error=f"HTTP {response.status}" 

                                    ) 

                            except Exception as e: 

                                self.report.add_assertion( 

                                    'Navigation', 

                                    'Navigation failed', 

                                    status='Failed', 

                                    url=page.url, 

                                    error=str(e) 

                                ) 

                        else: 

                            try: 

                                element.click() 

                                self.report.add_assertion( 

                                    'Interaction', 

                                    f'Successfully clicked {tag_name} element', 

                                    status='Passed', 

                                    url=page.url 

                                ) 

                                 

                                # Wait for any dynamic content 

                                try: 

                                    page.wait_for_load_state('networkidle', timeout=5000) 

                                    self.report.add_assertion( 

                                        'Dynamic Content', 

                                        'Page reached stable state after interaction', 

                                        status='Passed', 

                                        url=page.url 

                                    ) 

                                except Exception as e: 

                                    self.report.add_assertion( 

                                        'Dynamic Content', 

                                        'Page did not reach stable state after interaction', 

                                        status='Failed', 

                                        url=page.url, 

                                        error=str(e) 

                                    ) 

                            except Exception as e: 

                                self.report.add_assertion( 

                                    'Interaction', 

                                    f'Failed to click {tag_name} element', 

                                    status='Failed', 

                                    url=page.url, 

                                    error=str(e) 

                                ) 

                         

                        time.sleep(1)  # Small delay between actions 

                         

                    except Exception as e: 

                        self.report.add_assertion( 

                            'Test Step', 

                            'Failed to complete test step', 

                            status='Failed', 

                            url=page.url, 

                            error=str(e) 

                        ) 

                        continue 

                 

                print("\nTest execution completed!") 

                 

                # Final assertions 

                self.report.add_assertion( 

                    'Test Completion', 

                    f'Test completed successfully. Duration: {int(time.time() - start_time)} seconds', 

                    status='Passed', 

                    url=page.url 

                ) 

                 

                context.close() 

                browser.close() 

                 

                # Generate and save the HTML report 

                self.report.generate_html_report() 

                 

        except Exception as e: 

            self.report.add_assertion( 

                'Critical Error', 

                'Test execution failed with critical error', 

                status='Failed', 

                error=str(e) 

            ) 

            print(f"Error during test execution: {str(e)}") 