import pandas as pd
from IPython.display import display
import utils.file as file_utils
import utils.helpers as helpers


def create_and_transform_dataframes(debug=False):
    """
    Leest en transformeert alle benodigde dataframes.
    
    Args:
        debug (bool): Of debug output getoond moet worden
        
    Returns:
        tuple: (df, tasks_exploded_df) - Hoofddataframe en geëxplodeerde tasks
    """
    # Lees de data in
    df_requests = file_utils.lees_data()
    if debug:
        display(df_requests.tail())
    
    df_responses = file_utils.lees_responses()
    if debug:
        display(df_responses.tail())
    
    # Samenvoegen van df_requests en df_responses op route_id, date en time
    df = pd.merge(df_requests, df_responses, on=['route_id', 'date', 'time'], how='inner')
    if debug:
        display(df.tail())
    
    # Bereken totalen
    df['total_tasks'] = df['tasks'].apply(len)
    df['total_fixed_tasks'] = df['fixedTasks'].apply(len)
    df['total_tasks_sequence'] = df['tasks_sequence'].apply(len)
    
    # Converteer date naar datetime (van format YYYYMMDD naar datetime)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # Zet time kolom om naar int
    df['time'] = df['time'].astype(int)
    
    # Sorteer de data op date en time
    df = df.sort_values(['date', 'time'], ascending=[True, True])
    if debug:
        display(df.tail())
    
    # Filter data voor 18:00 uur
    df = df[df['time'] < 180000]
    if debug:
        display(df.tail())
    
    # Hernoem en verwijder kolommen
    df = df.rename(columns={
        'number_of_tasks_x': 'number_of_tasks',
        'number_of_tasks_in_input_plan_x': 'number_of_tasks_in_input_plan',
    })
    df = df.drop(columns=['number_of_tasks_y', 'number_of_tasks_in_input_plan_y'])
    
    if debug:
        print(df.columns.tolist())
    
    # Creëer tasks_exploded_df
    tasks_exploded_df = _create_tasks_exploded_df(df, debug)
    
    # Voeg route afstanden toe
    df = _add_route_distances(df, tasks_exploded_df)
    
    # Voeg extra analyse kolommen toe
    df = _add_analysis_columns(df)
    
    if debug:
        display(df[['route_id', 'date', 'time', 'total_tasks', 'total_tasks_added', 
                   'total_tasks_removed', 'route_distance_km', 'route_add_distance_km', 
                   'route_remove_distance_km']].tail(100))
    
    return df, tasks_exploded_df


def _create_tasks_exploded_df(df, debug=False):
    """
    Maakt een geëxplodeerd dataframe van alle tasks.
    """
    # Explodeer tasks
    tasks_exploded_df = df[['id', 'route_id', 'date', 'time', 'tasks']].explode('tasks').reset_index(drop=True)
    tasks_exploded_df = tasks_exploded_df.rename(columns={'tasks': 'task'})
    
    # Extraheer address en coördinaten
    tasks_exploded_df['address'] = tasks_exploded_df['task'].apply(
        lambda x: x.get('address') if isinstance(x, dict) else None
    )
    tasks_exploded_df['latitude'] = pd.to_numeric(
        tasks_exploded_df['address'].apply(
            lambda a: a.get('latitude') if isinstance(a, dict) else None
        ),
        errors='coerce'
    )
    tasks_exploded_df['longitude'] = pd.to_numeric(
        tasks_exploded_df['address'].apply(
            lambda a: a.get('longitude') if isinstance(a, dict) else None
        ),
        errors='coerce'
    )
    
    # Extraheer task_id
    tasks_exploded_df['task_id'] = tasks_exploded_df['task'].apply(
        lambda t: t.get('id') if isinstance(t, dict) else None
    )
    tasks_exploded_df.drop(columns=['address', 'task'], inplace=True, errors='ignore')
    
    # Genereer unieke location_id's
    tasks_exploded_df = helpers.generate_unique_location_id(tasks_exploded_df, 1.0)
    
    # Voeg volgorde toe
    tasks_exploded_df = _add_task_sequence(df, tasks_exploded_df)
    
    # Voeg is_fixed flag toe
    tasks_exploded_df = _add_fixed_flag(df, tasks_exploded_df)
    
    # Sorteer
    tasks_exploded_df = tasks_exploded_df.sort_values(
        ['date', 'time', 'route_id', 'volgorde'], 
        ascending=[True, True, True, True]
    ).reset_index(drop=True)
    
    if debug:
        display(tasks_exploded_df.tail())
    
    return tasks_exploded_df


def _add_task_sequence(df, tasks_exploded_df):
    """
    Voegt volgorde kolom toe op basis van tasks_sequence.
    """
    sequence_exploded = df[['id', 'route_id', 'date', 'time', 'tasks_sequence']].explode('tasks_sequence').reset_index(drop=True)
    sequence_exploded = sequence_exploded.rename(columns={'tasks_sequence': 'task_id'})
    sequence_exploded['volgorde'] = sequence_exploded.groupby(['id', 'route_id', 'date', 'time']).cumcount() + 1
    
    tasks_exploded_df = tasks_exploded_df.merge(
        sequence_exploded[['id', 'route_id', 'date', 'time', 'task_id', 'volgorde']],
        on=['id', 'route_id', 'date', 'time', 'task_id'],
        how='inner'
    )
    
    return tasks_exploded_df


def _add_fixed_flag(df, tasks_exploded_df):
    """
    Voegt is_fixed kolom toe op basis van fixedTasks.
    """
    fixed_exploded = df[['id', 'route_id', 'date', 'time', 'fixedTasks']].explode('fixedTasks').reset_index(drop=True)
    fixed_exploded = fixed_exploded.rename(columns={'fixedTasks': 'fixed_task'})
    fixed_exploded['task_id'] = fixed_exploded['fixed_task'].apply(
        lambda x: x.get('taskId') if isinstance(x, dict) and 'taskId' in x else x
    )
    fixed_exploded['is_fixed'] = True
    fixed_exploded.drop(columns=['fixed_task'], inplace=True, errors='ignore')
    
    tasks_exploded_df = tasks_exploded_df.merge(
        fixed_exploded[['id', 'route_id', 'date', 'time', 'task_id', 'is_fixed']],
        on=['id', 'route_id', 'date', 'time', 'task_id'],
        how='left'
    )
    
    tasks_exploded_df['is_fixed'] = tasks_exploded_df['is_fixed'].fillna(False).astype(bool)
    
    return tasks_exploded_df


def _add_route_distances(df, tasks_exploded_df):
    """
    Voegt route afstand kolommen toe.
    """
    # Bereken totaal aantal toegevoegde en verwijderde tasks
    df['total_tasks_added'] = (
        df['total_tasks'] - df.groupby(['route_id', 'date'])['total_tasks_sequence'].shift(1).fillna(0)
    ).clip(lower=0)
    df['total_tasks_removed'] = (
        df.groupby(['route_id', 'date'])['total_tasks_sequence'].shift(1).fillna(0) - df['total_tasks']
    ).clip(lower=0)
    
    # Voeg location lists toe
    df['locations_added_list'] = df.apply(lambda row: helpers.get_added_locations(row, tasks_exploded_df), axis=1)
    df['locations_removed_list'] = df.apply(lambda row: helpers.get_removed_locations(row, tasks_exploded_df), axis=1)
    
    # Bereken route afstanden
    dist_map = helpers.compute_route_distances_map(tasks_exploded_df)
    df['_snap_key'] = list(zip(df['id'], df['route_id'], df['date'], df['time']))
    df['route_distance_km'] = df['_snap_key'].map(dist_map).fillna(0.0)
    df.drop(columns=['_snap_key'], inplace=True)
    
    # Bereken toegevoegde en verwijderde afstanden
    df['route_add_distance_km'] = (
        df['route_distance_km'] - df.groupby(['route_id', 'date'])['route_distance_km'].shift(1).fillna(0)
    ).clip(lower=0)
    df['route_remove_distance_km'] = (
        df.groupby(['route_id', 'date'])['route_distance_km'].shift(1).fillna(0) - df['route_distance_km']
    ).clip(lower=0)
    
    return df


def _add_analysis_columns(df):
    """
    Voegt extra analyse kolommen toe.
    """
    df['diff_route_total_distance'] = df['route_add_distance_km'] - df['route_remove_distance_km']
    df['avg_km_per_task'] = df.apply(
        lambda row: row['route_distance_km'] / row['total_tasks'] if row['total_tasks'] > 0 else 0, 
        axis=1
    )
    df['total_tasks_moved'] = df['total_tasks_added'] + df['total_tasks_removed']
    
    return df
