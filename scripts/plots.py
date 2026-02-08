import matplotlib.pyplot as plt


def plot_total_tasks_per_day(df):
    """
    Plot het totaal aantal taken per dag voor input en output.
    
    Args:
        df: DataFrame met route data
    """
    # Groepeer op route_id, date en time en houdt enkel de eerste/laatste van de dag over
    tasks_per_day_input = (
        df.sort_values(['route_id', 'date', 'time'])
        .groupby(['route_id', 'date'])
        .first()
        .reset_index()
    ).groupby('date')['total_tasks'].sum().reset_index()
    
    tasks_per_day_output = (
        df.sort_values(['route_id', 'date', 'time'])
        .groupby(['route_id', 'date'])
        .last()
        .reset_index()
    ).groupby('date')['total_tasks'].sum().reset_index()
    
    # Plot input vs output
    plt.figure(figsize=(12, 6))
    plt.plot(tasks_per_day_input['date'], tasks_per_day_input['total_tasks'], marker='o', label='Input')
    plt.plot(tasks_per_day_output['date'], tasks_per_day_output['total_tasks'], marker='o', label='Output')
    plt.title('Totale Taken per Dag')
    plt.xlabel('Datum (date)')
    plt.ylabel('Totaal Aantal Taken')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def plot_total_routes_per_day(df):
    """
    Plot het totaal aantal routes per dag.
    
    Args:
        df: DataFrame met route data
    """
    # Aantal unieke routes per dag
    routes_per_day = (
        df.sort_values(['route_id', 'date', 'time'])
        .groupby(['route_id', 'date'])
        .first()
        .reset_index()
    ).groupby('date')['route_id'].count().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(routes_per_day['date'], routes_per_day['route_id'], marker='o')
    plt.title('Aantal Routes per Dag')
    plt.xlabel('Datum (date)')
    plt.ylabel('Aantal Routes')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def plot_average_tasks_per_routes_per_day(df):
    """
    Plot het gemiddeld aantal taken per route per dag.
    
    Args:
        df: DataFrame met route data
    """
    # Bereken gemiddeld aantal taken per route per dag
    daily_data = (
        df.sort_values(['route_id', 'date', 'time'])
        .groupby(['route_id', 'date'])
        .last()
        .reset_index()
    )
    
    avg_tasks_per_day = daily_data.groupby('date').agg({
        'total_tasks': 'mean',
        'route_id': 'count'
    }).reset_index()
    avg_tasks_per_day.columns = ['date', 'avg_tasks_per_route', 'num_routes']
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_tasks_per_day['date'], avg_tasks_per_day['avg_tasks_per_route'], marker='o')
    plt.title('Gemiddeld Aantal Taken per Route per Dag')
    plt.xlabel('Datum (date)')
    plt.ylabel('Gemiddeld Aantal Taken per Route')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def plot_average_km_per_route_per_day(df):
    """
    Plot het gemiddeld aantal kilometers per route per dag.
    
    Args:
        df: DataFrame met route data
    """
    # Bereken gemiddeld aantal km per route per dag
    daily_data = (
        df.sort_values(['route_id', 'date', 'time'])
        .groupby(['route_id', 'date'])
        .last()
        .reset_index()
    )
    
    avg_km_per_day = daily_data.groupby('date').agg({
        'route_distance_km': 'mean',
        'route_id': 'count'
    }).reset_index()
    avg_km_per_day.columns = ['date', 'avg_km_per_route', 'num_routes']
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_km_per_day['date'], avg_km_per_day['avg_km_per_route'], marker='o')
    plt.title('Gemiddeld Aantal Kilometers per Route per Dag')
    plt.xlabel('Datum (date)')
    plt.ylabel('Gemiddeld Aantal Kilometers')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def plot_average_movements_per_route_per_day(df):
    """
    Plot het gemiddeld aantal bewegingen (tasks moved) per route per dag.
    
    Args:
        df: DataFrame met route data
    """
    # Bereken gemiddeld aantal bewegingen per route per dag
    daily_data = (
        df.sort_values(['route_id', 'date', 'time'])
        .groupby(['route_id', 'date'])
        .agg({
            'total_tasks_moved': 'sum',
            'route_id': 'first'
        })
        .reset_index(drop=True)
    )
    daily_data['date'] = df.sort_values(['route_id', 'date', 'time']).groupby(['route_id', 'date'])['date'].first().values
    
    avg_movements_per_day = daily_data.groupby('date').agg({
        'total_tasks_moved': 'mean'
    }).reset_index()
    avg_movements_per_day.columns = ['date', 'avg_movements_per_route']
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_movements_per_day['date'], avg_movements_per_day['avg_movements_per_route'], marker='o')
    plt.title('Gemiddeld Aantal Bewegingen per Route per Dag')
    plt.xlabel('Datum (date)')
    plt.ylabel('Gemiddeld Aantal Bewegingen (Tasks Moved)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def plot_total_tasks_per_day_per_route(df, route_id):
    """
    Plot het totaal aantal taken per dag voor een specifieke route.
    
    Args:
        df: DataFrame met route data
        route_id: De route_id om te filteren
    """
    # Filter data voor de gegeven route_id
    route_data = df[df['route_id'] == route_id].copy()
    
    if route_data.empty:
        print(f"Geen data gevonden voor route_id: {route_id}")
        return
    
    # Groepeer per dag en neem de laatste snapshot van de dag
    tasks_per_day = (
        route_data.sort_values(['date', 'time'])
        .groupby('date')
        .last()
        .reset_index()
    )[['date', 'total_tasks']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(tasks_per_day['date'], tasks_per_day['total_tasks'], marker='o')
    plt.title(f'Totaal Aantal Taken per Dag voor Route {route_id}')
    plt.xlabel('Datum (date)')
    plt.ylabel('Totaal Aantal Taken')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def plot_total_tasks_per_day_per_route_from_begin_and_end(df, route_id):
    """
    Plot het totaal aantal taken per dag voor een specifieke route (begin en eind snapshot).
    
    Args:
        df: DataFrame met route data
        route_id: De route_id om te filteren
    """
    # Filter data voor de gegeven route_id
    route_data = df[df['route_id'] == route_id].copy()
    
    if route_data.empty:
        print(f"Geen data gevonden voor route_id: {route_id}")
        return
    
    # Groepeer per dag en neem zowel eerste als laatste snapshot
    tasks_per_day_first = (
        route_data.sort_values(['date', 'time'])
        .groupby('date')
        .first()
        .reset_index()
    )[['date', 'total_tasks']]
    tasks_per_day_first.columns = ['date', 'total_tasks_first']
    
    tasks_per_day_last = (
        route_data.sort_values(['date', 'time'])
        .groupby('date')
        .last()
        .reset_index()
    )[['date', 'total_tasks']]
    tasks_per_day_last.columns = ['date', 'total_tasks_last']
    
    # Merge de data
    tasks_per_day = tasks_per_day_first.merge(tasks_per_day_last, on='date')
    
    plt.figure(figsize=(12, 6))
    plt.plot(tasks_per_day['date'], tasks_per_day['total_tasks_first'], marker='o', label='Start planning van de dag')
    plt.plot(tasks_per_day['date'], tasks_per_day['total_tasks_last'], marker='s', label='Einde planning van de dag')
    plt.title(f'Totaal Aantal Taken per Dag voor Route {route_id} (Begin vs Einde)')
    plt.xlabel('Datum (date)')
    plt.ylabel('Totaal Aantal Taken')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
