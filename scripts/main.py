import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import ipywidgets as widgets
from config import DATA_DIR
from create_dataframes import create_and_transform_dataframes
from plots import (
    plot_total_tasks_per_day, 
    plot_total_routes_per_day, 
    plot_average_tasks_per_routes_per_day,
    plot_average_km_per_route_per_day,
    plot_average_movements_per_route_per_day,
    plot_total_tasks_per_day_per_route,
    plot_total_tasks_per_day_per_route_from_begin_and_end
)
import pandas as pd
from styling import standard_style
standard_style()

debug = True

# Creëer en transformeer alle dataframes
df, tasks_exploded_df = create_and_transform_dataframes(debug=debug)
display(df.info())
display(df.head())
# Plot totale taken per dag
plot_total_tasks_per_day(df)

# Plot totale routes per dag
plot_total_routes_per_day(df)

# Plot gemiddelde taken per route per dag
plot_average_tasks_per_routes_per_day(df)

# Plot gemiddelde aantal km per route per dag
plot_average_km_per_route_per_day(df)

# Plot gemiddelde aantal bewegingen per route per dag
plot_average_movements_per_route_per_day(df)

# Console-gebaseerde route selectie
unique_routes = sorted(df['route_id'].unique().tolist())

print("\n=== Beschikbare Routes ===")
for idx, route_id in enumerate(unique_routes, 1):
    print(f"{idx}. {route_id}")

while True:
    try:
        choice = int(input(f"\nKies een route nummer (1-{len(unique_routes)}): "))
        if 1 <= choice <= len(unique_routes):
            selected_route = unique_routes[choice - 1]
            print(f"\nGeselecteerde route: {selected_route}")
            
            plot_total_tasks_per_day_per_route(df, selected_route)
            plot_total_tasks_per_day_per_route_from_begin_and_end(df, selected_route)
            break
        else:
            print("Ongeldig nummer, probeer opnieuw.")
    except ValueError:
        print("Voer een geldig nummer in.")

