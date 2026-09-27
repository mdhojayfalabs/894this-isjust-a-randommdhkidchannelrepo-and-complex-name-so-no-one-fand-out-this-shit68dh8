#!/usr/bin/env python3
"""ONE-TIME SETUP - run on YOUR computer (not on GitHub).
Steps:
  1. pip install -r requirements.txt
  2. python scripts/setup_auth.py path/to/client_secret.json
  3. A browser window opens -> log in with the Gmail that owns the channel
     -> allow access
  4. Copy the three values it prints into GitHub repo Secrets:
       YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN

NEVER share these values with anyone (not even with any AI/chat tool).
"""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_auth.py client_secret.json")
        sys.exit(1)
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    scope = "https://www.googleapis.com/auth/youtube.upload"
    flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1], [scope])
    creds = flow.run_local_server(port=0)  # opens browser, prints callback locally
    assert creds.refresh_token, "refresh token missing - retry"
    print("\n================ COPIE THESE INTO GITHUB SECRETS ================")
    print(f"YT_CLIENT_ID     = {creds.client_id}")
    print(f"YT_CLIENT_SECRET = {creds.client_secret}")
    print(f"YT_REFRESH_TOKEN = {creds.refresh_token}")
    print("====================================================================")


if __name__ == "__main__":
    main()
