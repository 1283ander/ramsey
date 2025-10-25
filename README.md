# Drone SRT Processor

A Python application for processing drone flight data and SRT (SubRip) subtitle files from DJI drones.

## Project Structure

- `drone_srt_processor.py` - Main Python script for processing SRT files
- `Routes/` - Directory containing flight data and routes
  - `*.SRT` - Individual SRT subtitle files from drone flights
  - `all_flights.json` - Metadata for all flights
  - `*_flight_path.kml` - KML files containing flight path data
  - `flights/` - Subdirectory with additional flight data in KML format

## Features

- Process SRT files from DJI drone footage
- Extract flight telemetry data (altitude, speed, coordinates)
- Generate flight path visualizations
- Analyze flight patterns and statistics

## Usage

Run the main processor script:

```bash
python drone_srt_processor.py
```

## Git Setup

This repository is configured with:
- Main branch (modern standard)
- Comprehensive .gitignore for Python projects
- All drone data files tracked (SRT, KML, JSON)
- Video files excluded from version control

## Requirements

- Python 3.x
- Standard library modules (no external dependencies required)

## Flight Data

The project includes flight data from multiple drone missions:
- SRT files with timestamped telemetry data
- KML flight path files for geospatial visualization
- JSON metadata for flight organization
