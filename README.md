# Anigui App

Anigui is a desktop anime client built with **PySide6** and **anipy-api**.
It includes search, episode playback, watchlist/history management, favorites, downloads, offline browsing, and AniList sync features.

## Features

- Anime search with provider/language/quality selection
- Episode list and integrated player controls
- Watchlist with status sections (`Watching`, `Planned`, `Completed`)
- Resume playback from saved progress
- Favorites management
- Download queue with parallel tasks
- Offline library view for downloaded episodes
- Optional AniList sync/import
- UI localization (Italian / English)

## Platform support

- **Windows**: recommended via prebuilt release executable
- **Linux**: run directly from source (`app.py`)

## Requirements

- Python **3.10+**
- `pip`
- Network access for providers and AniList (if enabled)

## Installation

### Windows (from source)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux (from source)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Windows

```powershell
python app.py
```

### Linux

```bash
python3 app.py
```

## Configuration and data

User data is stored in the OS application state folder:

- Watch history
- Search history
- Offline cover map
- Favorites
- Metadata cache
- Settings (including AniList token if enabled)

The app manages these files automatically at runtime.

## AniList

AniList integration is optional. If enabled:

- Set your Personal Access Token in Settings
- Use `Test AniList connection` to validate credentials
- Use sync/import actions from the Settings tab

### Quick setup (recommended)

1. Open **AniList -> Settings -> Developer -> Create New Client**
2. Set **Redirect URI** to:
   `https://anilist.co/api/v2/oauth/pin`
3. Save and copy your `client_id`
4. Open in browser:
   `https://anilist.co/api/v2/oauth/authorize?client_id=YOUR_CLIENT_ID&response_type=token`
5. Authorize the app
6. Copy the `access_token` from the resulting page/URL
7. Paste it into this app: **Settings -> AniList token**

## Security notes

- Do not commit personal tokens or credentials
- Keep local settings files private on shared machines
- External provider endpoints may change or become unavailable

## Troubleshooting

- If playback fails, verify `mpv` availability on your system
- If a provider returns empty/failed results, retry later or switch provider
- If AniList sync fails, verify token validity and network connectivity

## Project scope

This repository is focused on the application source and runtime behavior.
Windows binaries are distributed separately through releases.
