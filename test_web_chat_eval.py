import pytest
from playwright.sync_api import sync_playwright

# 1. Representative Chat UI Test Function
def test_ui_chatbot_response():
    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load a public demo web page
        page.goto("https://example.com")
        
        # Extract title as simulated chatbot output
        page_title = page.title()
        browser.close()

        # QA Guardrail: Verify web page response content is non-empty and accurate
        assert page_title == "Example Domain", f"Unexpected web UI output: {page_title}"

# 2. Automated Chatbot Payload Extraction Test
def test_chat_payload_extraction():
    # Simulated UI input/output pair scraped from a web chatbot
    user_prompt = "How do I update my profile?"
    bot_ui_response = "To update your profile, click on your avatar in the top right and select Edit Profile."

    # Verify UI response contains critical action keywords
    required_keywords = ["profile", "edit", "avatar"]
    
    assert all(kw in bot_ui_response.lower() for kw in required_keywords), \
        "UI response missing core user navigation instructions."