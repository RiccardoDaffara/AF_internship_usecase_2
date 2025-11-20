import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration for file paths
DATA_DIR = 'files_usecase2'
OUTPUT_FILE = 'AF-KLM_flights_with_BFL.csv.gz'


def load_and_clean_capacities(filepath):
    """
    Loads capacities data, handles parsing errors, fixes data types,
    and corrects specific data entry errors.
    """
    print("--- Loading Capacities ---")
    # 1. Get valid column names to avoid ParserError on bad lines
    col_names = pd.read_csv(filepath, sep=';', nrows=0).columns.tolist()

    # 2. Read file using specific columns
    df = pd.read_csv(filepath, sep=';', usecols=col_names)

    # 3. Fix data types
    df['seats'] = df['seats'].astype('Int64')
    df['pax'] = df['pax'].astype('Int64')

    # 4. Fix specific data entry error (hardcoded based on exploration)
    # Row index 83 had a textual date
    if len(df) > 83:
        df.at[83, 'date'] = '2025-12-01'

    # 5. Convert date
    df['date'] = pd.to_datetime(df['date'])

    # 6. Calculate Load Factor
    df['BLF'] = ((df['pax'] / df['seats']) * 100).round(2)

    return df


def load_and_clean_revenues(filepath):
    """Loads revenue data and converts dates."""
    print("--- Loading Revenues ---")
    df = pd.read_csv(filepath, sep='\t')
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    return df


def load_and_clean_flights(filepath):
    """Loads flight details and reconstructs full dates."""
    print("--- Loading Flights ---")
    headers = ['flight_number', 'carrier', 'origin', 'destination', 'plane', 'departure_date']

    # Note: index_col=0 implies the first column in file is the index
    df = pd.read_csv(filepath, sep='\t', names=headers)

    # Add year '2025' to the date string before converting
    df['departure_date'] = pd.to_datetime('2025' + df['departure_date'], format='%Y-%m-%d')

    return df


def merge_datasets(df_cap, df_rev, df_flights):
    """Merges the three datasets into a single analytical dataframe."""
    print("--- Merging Datasets ---")

    # Merge Capacities + Revenues
    df_temp = pd.merge(
        df_cap,
        df_rev,
        left_on=['flight_number', 'date'],
        right_on=['FlightNumber', 'flight_date'],
        how='inner'
    )

    # Merge Result + Flights
    df_final = pd.merge(
        df_temp,
        df_flights,
        left_on=['flight_number', 'date'],
        right_on=['flight_number', 'departure_date'],
        how='inner'
    )

    # Clean up duplicate/unnecessary columns
    cols_to_drop = ['FlightNumber', 'flight_date', 'departure_date']
    df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Reorder columns for readability
    final_columns = ['flight_number', 'date', 'seats', 'pax', 'revenue', 'BLF',
                     'carrier', 'origin', 'destination', 'plane']

    return df_final[final_columns]


def print_key_metrics(df_final, df_revenues_raw):
    """Calculates and prints key business metrics."""
    print("\n--- Key Metrics Report ---")

    # 1. Average PAX AMS-JFK
    mask_route = ((df_final['origin'] == 'AMS') & (df_final['destination'] == 'JFK')) | \
                 ((df_final['origin'] == 'JFK') & (df_final['destination'] == 'AMS'))
    avg_pax = df_final.loc[mask_route, 'pax'].mean()
    print(f"Avg Pax (AMS <-> JFK): {avg_pax:.2f}")

    # 2. Carrier Revenues
    rev_af = df_final.loc[df_final['carrier'] == 'AF', 'revenue'].sum()
    rev_kl = df_final.loc[df_final['carrier'] == 'KL', 'revenue'].sum()
    total_merged = rev_af + rev_kl

    print(f"Total Revenue AF: {rev_af:,.2f} €")
    print(f"Total Revenue KL: {rev_kl:,.2f} €")
    print(f"Total Revenue (Merged): {total_merged:,.2f} €")

    # 3. Data Integrity Check
    total_raw = df_revenues_raw['revenue'].sum()
    diff = total_raw - total_merged

    if diff > 0.01:  # accounting for float precision
        print(f"\n[Notice] Revenue Discrepancy found: {diff:,.2f} €")
        # Identify non-AF/KL flights (likely code shares or other carriers)
        mask_af_kl = df_revenues_raw['FlightNumber'].str.startswith(('A', 'K'))
        other_rev = df_revenues_raw[~mask_af_kl]['revenue'].sum()
        print(f"Revenue from non-AF/KL flights (e.g., DL): {other_rev:,.2f} €")
    else:
        print("Revenue matches perfectly.")


def generate_plots(df_final):
    """Generates and saves visualization plots."""
    print("\n--- Generating Plots ---")

    # Setup
    sns.set_theme(style="whitegrid")

    # Plot 1: Daily Revenue Evolution
    daily_revenue = df_final.groupby(df_final['date'].dt.date)['revenue'].sum().reset_index()
    daily_revenue.columns = ['Day', 'Total Revenue']
    mean_rev = daily_revenue['Total Revenue'].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_revenue, x='Day', y='Total Revenue', marker='o', color='darkblue')
    plt.axhline(y=mean_rev, color='red', linestyle='--', alpha=0.5, label=f'Mean ({mean_rev:,.0f} €)')
    plt.title("Daily Flight Revenue Evolution")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_daily_revenue.png')
    print("Saved: plot_daily_revenue.png")
    plt.close()

    # Plot 2: Top 10 Profitable Routes
    route_revenue = df_final.groupby(['origin', 'destination'])['revenue'].sum().reset_index()
    route_revenue['Route'] = route_revenue['origin'] + ' - ' + route_revenue['destination']
    top_routes = route_revenue.sort_values(by='revenue', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='revenue', y='Route', data=top_routes, hue='Route', palette='viridis', legend=False)
    plt.title("Top 10 Most Profitable Routes")
    plt.xlabel("Total Revenue (€)")
    plt.tight_layout()
    plt.savefig('plot_top_routes.png')
    print("Saved: plot_top_routes.png")
    plt.close()


def main():
    # 1. Load Data
    try:
        df_cap = load_and_clean_capacities(f'{DATA_DIR}/capacities.txt')
        df_rev = load_and_clean_revenues(f'{DATA_DIR}/revenues.bz2')
        df_flights = load_and_clean_flights(f'{DATA_DIR}/flights.gz')
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return

    # 2. Merge Data
    df_final = merge_datasets(df_cap, df_rev, df_flights)

    # 3. Export Clean Data
    print(f"--- Exporting to {OUTPUT_FILE} ---")
    df_final.to_csv(OUTPUT_FILE, index=False, compression='gzip')

    # 4. Reporting
    print_key_metrics(df_final, df_rev)

    # 5. Visualization
    generate_plots(df_final)

    print("\nProcess Completed Successfully.")


if __name__ == "__main__":
    main()