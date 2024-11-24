from playwright.sync_api import sync_playwright
import json
from datetime import datetime
import time

class InteractionRecorder:
    def __init__(self):
        self.interactions = []
        
    def start_recording(self, url, duration=60):
        print(f"\nStarting interaction recording for {url}")
        print(f"The browser will open and record your interactions for {duration} seconds.")
        print("Please start interacting with the website...\n")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()
                page = context.new_page()
                
                # Define event handlers with proper event data extraction
                def handle_click(event):
                    print("Click detected!")
                    self._record_event("click", {
                        "selector": event.get("selector", ""),
                        "x": event.get("position", {}).get("x", 0),
                        "y": event.get("position", {}).get("y", 0),
                        "url": page.url
                    })

                def handle_input(event):
                    print("Input detected!")
                    self._record_event("input", {
                        "selector": event.get("selector", ""),
                        "value": event.get("value", ""),
                        "url": page.url
                    })

                def handle_navigation(event):
                    print("Navigation detected!")
                    self._record_event("navigation", {
                        "url": page.url
                    })

                # Attach event listeners with proper handlers
                page.on("click", handle_click)
                page.on("input", handle_input)
                page.on("framenavigated", handle_navigation)
                
                # Navigate to URL
                print("Opening browser...")
                page.goto(url)
                print("Recording started. You have {} seconds to perform your interactions.".format(duration))
                
                # Record for specified duration
                start_time = time.time()
                while time.time() - start_time < duration:
                    remaining = int(duration - (time.time() - start_time))
                    if remaining % 10 == 0:  # Print remaining time every 10 seconds
                        print(f"{remaining} seconds remaining...")
                        print(f"Current interaction count: {len(self.interactions)}")
                    time.sleep(1)
                
                print("\nRecording completed!")
                print(f"Total interactions recorded: {len(self.interactions)}")
                context.close()
                browser.close()
                    
        except Exception as e:
            print(f"An error occurred during recording: {str(e)}")
            raise
            
    def _record_event(self, event_type, data):
        try:
            interaction = {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                **data
            }
            self.interactions.append(interaction)
            print(f"Recorded {event_type} interaction: {json.dumps(interaction, indent=2)}")
        except Exception as e:
            print(f"Error recording {event_type} event: {str(e)}")
        
    def save_interactions(self, filename):
        if not self.interactions:
            print("Warning: No interactions were recorded!")
        with open(filename, 'w') as f:
            json.dump(self.interactions, f, indent=2)
            print(f"Saved {len(self.interactions)} interactions to {filename}")