#!/usr/bin/env python3
"""ONE-TIME SETUP - run on YOUR device (not on GitHub Actions).
Desktop:
  1. pip install -r requirements.txt
  2. python scripts/setup_auth.py path/to/client_secret.json
  3. A browser window opens -> log in with the Gmail that owns the channel
     -> allow access
Phone / no browser attached (e.g. GitHub Codespaces):
  2. python scripts/setup_auth.py client.json --console
  3. Copy the URL it prints -> open it in your phone browser -> log in with
     the Gmail that owns the channel -> Allow
  4. The browser shows an error page (normal - it is a local callback).
     Copy the FULL URL from the address bar and paste it back in the terminal.
  5. The three values it prints go to the agent, which installs them
     encrypted into repo Secrets + vault (never paste them anywhere public).
"""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_auth.py client_secret.json [--console]")
        sys.exit(1)
    from google_auth_oauthlib.flow import InstalledAppFlow

    scope = "https://www.googleapis.com/auth/youtube.upload"
    flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1], [scope])
    if "--console" in sys.argv:
        print("\nOPEN THIS URL IN YOUR PHONE BROWSER NOW:")
        creds = flow.run_console()
    else:
        creds = flow.run_local_server(port=0)
    assert creds.refresh_token, "refresh token missing - retry"
    print("\n================ COPIE THESE AND SEND TO AGENT ================")
    print(f"YT_CLIENT_ID     = {creds.client_id}")
    print(f"YT_CLIENT_SECRET = {creds.client_secret}")
    print(f"YT_REFRESH_TOKEN = {creds.refresh_token}")
    print("====================================================================")


if __name__ == "__main__":
    main()
