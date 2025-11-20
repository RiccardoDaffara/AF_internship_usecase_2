# AF-KLM Data Analysis Revenues for the first week of December 2025

This script processes flight data for Air France and KLM to calculate revenue, load factors, and generate visualization plots for the first week of December 2025.

## 1. Setup

Make sure you have **Python 3** installed. Then, install the required libraries:

```bash
pip install pandas matplotlib seaborn
````

## 2\. How to Run

1.  Place your data files (`capacities.txt`, `revenues.bz2`, `flights.gz`) inside a folder named `files_usecase2`.
2.  Run the script from your terminal:

<!-- end list -->

```bash
python main.py
```

## 3\. Results

The script will create the following files in the same folder:

  * **`AF-KLM_flights_analysis.csv.gz`**: The final cleaned dataset.
  * **`plot_daily_revenue.png`**: A graph of revenue over time.
  * **`plot_top_routes.png`**: A chart of the most profitable routes.

<!-- end list -->