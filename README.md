# Floorball Match Prediction & Data Pipeline

> Personal data engineering and machine learning project focused on **floorball match data**, **player statistics scraping**, **feature engineering**, and **match outcome prediction**.

## Authors

`Author: Pavel Halík`

## Project Overview

This project collects, processes, and unifies data from multiple floorball competitions and player-stat sources. The goal is to build a clean dataset for analysis and machine learning, including:

* **match-level historical results**
* **official season standings**
* **player season statistics**
* **pre-match engineered features**
* **model training and prediction experiments**

The pipeline combines API-based collection, browser scraping, dataset standardization, and downstream modeling.

## Main Goals

* Download historical floorball match data across selected competitions
* Build a processed dataset with **pre-match features only**
* Scrape player statistics from multiple league sources
* Standardize raw player files into a unified schema
* Export official regular-season standings
* Prepare datasets for model training and prediction

## Technologies Used

* Python
* pandas
* requests
* Playwright
* Jupyter Notebook
* CSV-based data pipeline
* dotenv / environment configuration

## Data Pipeline

### 1. Match Collection

The match collection pipeline downloads season and match summary data from the configured API.

Main outputs:

* raw match dataset
* processed match dataset
* official standings dataset

Typical workflow:

1. fetch competition seasons
2. fetch season summaries
3. parse finished matches
4. compute rolling and table-based pre-match features
5. export official regular-season standings

### 2. Player Scraping

Player statistics are scraped from league-specific sources such as:

* F-Liiga
* SSL
* Extraliga

These scrapers export raw CSV files into the `data/raw/players` directory.

### 3. Player Standardization

Raw player CSV files are normalized into a unified schema with columns such as:

* `source`
* `league`
* `season`
* `team_name`
* `player_name`
* `gp`
* `goals`
* `assists`
* `points`
* `shots`
* `pim`
* `plus_minus`

### 4. Team Name Unification

Because team names differ between sources, the project includes manual mapping and normalization logic for:

* Czech competitions
* Finland competitions
* Sweden competitions

This helps connect player data, standings, and match data reliably.

### 5. Feature Engineering

The processed match dataset includes pre-match features such as:

* team rank before match
* points before match
* games played before match
* goal difference
* recent form over last matches
* home-only and away-only form
* target match result labels

## Example Output Datasets

### Raw Match Dataset

Contains parsed finished matches such as:

* competition id and name
* season id and season name
* match id
* start time
* home and away teams
* final score

### Processed Match Dataset

Contains machine-learning-ready rows with pre-match features and labels.

### Official Standings Dataset

Contains official regular-season standings per team, season, and competition.

### Unified Player Dataset

Contains standardized player-season statistics merged from multiple sources.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Forkxel/Floorball-Match-Predictor.git
cd floorball
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For Playwright run:

```bash
playwright install
```

### 4. Environment Configuration

Fill a API key in `.env` file in the project root.

Example:

```env
SPORTRADAR_API_KEY=your_api_key_here
```

## Configuration

The project uses configuration values from `src/collection/config.py` and `.env`.

Typical configured values include:

* API key
* base API URL
* format
* competition ids
* raw dataset path
* processed dataset path
* official standings path

## How to Run

### Collect Matches and Build Datasets

Run the main collection script:

```bash
python src/collection/collect_matches.py
```

This should:

* download seasons
* download match summaries
* parse finished matches
* save raw dataset
* save processed dataset
* save official standings

### Scrape Player Statistics

Examples:

```bash
python src/collection/scraping_players/scrape_fliiga_players.py
python src/collection/scraping_players/scrape_ssl_players.py
python src/collection/scraping_players/scrape_extraliga_players.py
```

### Merge and Standardize Player Data

```bash
python src/processing/merge_player_stats.py
```

### Additional Processing

```bash
python src/processing/merge_roster_to_matches.py
python src/processing/team_roster_strength.py
```

### Model Training / Experiments

Use:

* `src/learning_model/Floorball_model.ipynb`

## Application (UI)

The project also includes a simple application layer located in src/app/app.py.

This application provides a basic interface for interacting with the processed data and services defined in the project.

### Executable Version

A compiled executable version of the application is available in the `dist/` directory:

- Built using tools such as `PyInstaller`
- Can be run without installing Python or dependencies


## Key Implementation Details

### Match Parsing

The project parses only finished matches with valid:

* home team
* away team
* final score
* season and competition context

### Ranking Logic

Ranking is computed dynamically from historical match state before each match, so features reflect only information available before kickoff.

### Recent Form Features

Recent form is computed using rolling match history, including:

* overall last matches
* home-only history
* away-only history

### Standings Extraction

Official standings are fetched separately from season standings endpoints and filtered to regular-season standings only.

### Data Quality Handling

The pipeline includes helper utilities for:

* text cleaning
* numeric parsing
* season normalization
* team name normalization
* column alias matching


## Notes

* Some scraping logic depends on external website structure and may require updates if league websites change.
* API rate limits are handled with retry/wait logic where necessary.


## Contact

If you need anything from me about this application contact me at:

* pavel.halik06@gmail.com
