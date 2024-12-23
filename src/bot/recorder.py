from datetime import datetime
import json
import time
from playwright.sync_api import sync_playwright


class InteractionRecorder:
    def __init__(self):
        self.interactions = []

    def start_recording(self, url, duration=60):
        """
        Start recording user interactions on a webpage.

        Args:
            url (str): The URL to record interactions from
            duration (int): Duration in seconds to record

        Returns:
            list: List of recorded interactions
        """
        print(f"\nStarting interaction recording for {url}")
        print(f"The browser will open and record your interactions for {duration} seconds.")
        print("Please start interacting with the website...\n")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()
                page = context.new_page()

                # Reset interactions for new recording
                self.interactions = []
                self.last_url = None

                # Define event handlers with proper event data extraction
                def handle_click(event):
                    if "selector" in event:
                        print(f"Click detected on: {event['selector']}")
                        self._record_event(
                            "click",
                            {
                                "selector": event["selector"],
                                "x": event.get("position", {}).get("x", 0),
                                "y": event.get("position", {}).get("y", 0),
                                "url": page.url,
                            },
                        )

                def handle_input(event):
                    if "selector" in event:
                        print(f"Input detected on: {event['selector']}")
                        # Mask any sensitive input data
                        value = "***" if "password" in event.get("selector", "").lower() else event.get("value", "")
                        self._record_event(
                            "input",
                            {
                                "selector": event["selector"],
                                "value": value,
                                "url": page.url,
                            },
                        )

                def handle_navigation(url):
                    # Only record navigation if URL actually changed
                    if url != self.last_url:
                        print(f"Navigation detected to: {url}")
                        self._record_event("navigation", {"url": url})
                        self.last_url = url

                # Attach event listeners with proper handlers
                page.on("click", handle_click)
                page.on("input", handle_input)
                page.on("framenavigated", lambda frame: handle_navigation(frame.url))

                # Navigate to URL
                print("Opening browser...")
                page.goto(url)
                print(f"Recording started. You have {duration} seconds to perform your interactions.")

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

                # Return recorded interactions
                return self.interactions if self.interactions else None

        except Exception as e:
            print(f"An error occurred during recording: {str(e)}")
            return None

    def _record_event(self, event_type, data):
        """
        Record a single interaction event.
        """
        try:
            interaction = {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                **data,
            }
            self.interactions.append(interaction)
            print(f"Recorded {event_type} interaction: {json.dumps(interaction, indent=2)}")
        except Exception as e:
            print(f"Error recording {event_type} event: {str(e)}")

    def save_interactions(self, filename):
        """
        Save recorded interactions to a file.
        """
        if not self.interactions:
            print("Warning: No interactions were recorded!")
            return

        with open(filename, "w") as f:
            json.dump(self.interactions, f, indent=2)
            print(f"Saved {len(self.interactions)} interactions to {filename}")