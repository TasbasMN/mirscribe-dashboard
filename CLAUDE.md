# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands
- Run dashboard: `poetry run streamlit run dashboard.py`
- Install dependencies: `poetry install`
- Add dependency: `poetry add <package>`

## Code Style Guidelines
- Python 3.10+ compatible code
- Use type hints where appropriate
- Follow PEP 8 conventions for formatting
- Organize imports: standard library, third-party, local
- Variable naming: snake_case for variables and functions
- Error handling: use try/except with specific exceptions
- Function documentation: use docstrings for public functions

## Project Structure
- dashboard.py: Main Streamlit application
- data/: Contains data files (e.g., triplets.csv)
- Use Streamlit components for UI elements
- Prefer Pandas for data manipulation
- Use matplotlib/seaborn for visualizations