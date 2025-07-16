# Predicting NBA Careers - Basketball Data Science Project


> Authors: 
>  - Rasheed Martin
>  - Justin Cox
>  - Juan Sanchez 
>  - Matthew Wissell

<hr> 
Project Overview

This project aims to analyze basketball player data and build predictive models to forecast NBA career outcomes. Using a variety of data science techniques, we explore player performance metrics, historical data, and other relevant features to understand factors that influence career trajectories in professional basketball.

## Project File Structure:

```
├── data
│   ├── clean_data        # Processed and cleaned datasets ready for analysis
│   └── raw_data          # Original, unprocessed datasets
├── visualizations        # Plots, charts, and other visual outputs
├── notebooks             # Jupyter notebooks containing exploratory data analysis and modeling
├── utils                 # Utility scripts and helper functions
├── app.py                # Main application file for deploying models or visualizations
├── README.md             # Project documentation (this file)
└── requirements.txt      # Project dependencies
```

## Getting Started

### Prequisites

- Python 3.8+
- Jupyter Notebook (for running notebooks)
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```
   git clone <repo-url>
   ```
2. Create and activate a virtual environment

    ```
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
    
    ```
   pip install -r requirements.txt
   ```
## Running the Streamlit App
You can launch the interactive NBA career prediction app using Streamlit:
```
streamlit run app.py
```
This will open the app in your default web browser, where you can explore predictions and visualizations.


## Running with Docker
If you prefer to run the app in a Docker container, follow these steps:
1. Build the Docker image:
```
docker build -t nba-career-predictor .
```
2. Run the Docker container:
```
docker run -p 8501:8501 nba-career-predictor
```
3. Access the app:
Open your web browser and go to http://localhost:8501


## Usage

Explore the data and modeling approaches inside the notebooks directory.
Use app.py to launch the interactive application or model interface.
## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

Specify your license here (e.g., MIT License).