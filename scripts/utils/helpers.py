import pandas as pd
import numpy as np
from collections import Counter

from IPython.display import display

# Module-level cache for snapshot -> location_id list mapping
_cached_tasks_ref = None
_snapshot_locations_map = None
_snapshot_prev_key_map = None

def generate_unique_location_id(df, threshold_m: float = 1.0) -> pd.DataFrame:
    """Assign a compact `location_id` per unique rounded coordinate.

    This uses rounding to ~1m precision (5 decimals) and factorization to
    assign stable IDs quickly. It avoids expensive pairwise geodesic checks.

    Args:
        df: DataFrame with `latitude` and `longitude` columns.
        threshold_m: kept for API-compatibility but not used (rounding is used).

    Returns:
        DataFrame copy with an added `location_id` column (None when coords missing).
    """
    df = df.copy()

    # Mark missing coordinates
    missing_mask = df['latitude'].isna() | df['longitude'].isna()

    # Build coordinate keys: None for missing, else rounded tuple
    coord_keys = [None if missing else (round(lat, 5), round(lon, 5))
                  for lat, lon, missing in zip(df['latitude'], df['longitude'], missing_mask)]

    # Factorize via Series to avoid future warnings
    coord_series = pd.Series(coord_keys, dtype=object)
    codes, uniques = pd.factorize(coord_series)

    # Map codes to location ids, keep None for missing
    location_ids = [None if key is None else f"LOC_{code+1:03d}" for key, code in zip(coord_keys, codes)]

    df['location_id'] = location_ids
    return df


def get_added_tasks(tasks, tasks_sequence):
    """
    Return task dicts that appear in `tasks_sequence` but not in `tasks`.

    Accepts lists or pandas Series/ndarray and handles missing values.
    """
    # Normalize array-like inputs to lists
    if isinstance(tasks_sequence, (pd.Series, np.ndarray)):
        tasks_sequence = tasks_sequence.tolist()
    if tasks_sequence is None or (hasattr(tasks_sequence, '__len__') and len(tasks_sequence) == 0):
        return []

    original_task_ids = set()
    if isinstance(tasks, (pd.Series, np.ndarray)):
        tasks = tasks.tolist()
    if tasks:
        for task in tasks:
            if isinstance(task, dict) and 'id' in task:
                original_task_ids.add(task['id'])

    added_tasks = []
    for sequence_task in tasks_sequence:
        if isinstance(sequence_task, dict) and 'id' in sequence_task and sequence_task['id'] not in original_task_ids:
            added_tasks.append(sequence_task)

    return added_tasks


def get_added_locations(row, tasks: pd.DataFrame):
    """
    Return the list of `location_id`s that are present in the *next* snapshot
    for the same `(route_id, date)` but not in the current snapshot.

    To avoid expensive per-row DataFrame queries we cache maps built from
    `tasks`: one mapping snapshot key -> location_id list and another mapping
    snapshot key -> next snapshot key. The function returns a plain list.
    """
    global _cached_tasks_ref, _snapshot_locations_map, _snapshot_prev_key_map

    if row.get('total_tasks_added', 0) <= 0:
        return []

    # Rebuild cache if tasks DataFrame changed
    tasks_ref = (id(tasks), tasks.shape)
    if _cached_tasks_ref != tasks_ref or _snapshot_locations_map is None or _snapshot_prev_key_map is None:
        # Group by snapshot key and collect non-null location_ids once
        grp = tasks.groupby(['id', 'route_id', 'date', 'time'])['location_id'].apply(
            lambda s: [v for v in s.tolist() if pd.notna(v)]
        )
        # grp keys are tuples (id, route_id, date, time)
        _snapshot_locations_map = grp.to_dict()

        # Build previous-key mapping per (route_id, date) ordered by time
        snaps = tasks[['id', 'route_id', 'date', 'time']].drop_duplicates()
        _snapshot_prev_key_map = {}
        for (route, date), group in snaps.groupby(['route_id', 'date']):
            group_sorted = group.sort_values('time')
            ids = group_sorted['id'].tolist()
            times = group_sorted['time'].tolist()
            # map each snapshot to the previous snapshot key (None for first)
            for i, row_id in enumerate(ids):
                if i-1 >= 0:
                    prev_id = ids[i-1]
                    prev_time = times[i-1]
                    prev_key = (prev_id, route, date, prev_time)
                else:
                    prev_key = None
                key = (row_id, route, date, times[i])
                _snapshot_prev_key_map[key] = prev_key

        _cached_tasks_ref = tasks_ref

    key = (row.get('id'), row.get('route_id'), row.get('date'), row.get('time'))
    locations_current = _snapshot_locations_map.get(key, [])
    prev_key = _snapshot_prev_key_map.get(key)
    if prev_key:
        locations_prev = _snapshot_locations_map.get(prev_key, [])
    else:
        locations_prev = []

    # compute locations present in current snapshot but not in previous
    # Preserve the order from the current snapshot and include multiplicity
    prev_counts = Counter(locations_prev)
    added_locations = []
    for loc in locations_current:
        if prev_counts.get(loc, 0) > 0:
            prev_counts[loc] -= 1
            continue
        added_locations.append(loc)
    return added_locations


def get_removed_locations(row, tasks: pd.DataFrame):
    """
    Return the list of `location_id`s that were present in the *previous*
    snapshot for the same `(route_id, date)` but are not present in the
    current snapshot.

    Preserves the order from the previous snapshot and preserves multiplicity.
    Uses the same module-level cache as `get_added_locations`.
    """
    global _cached_tasks_ref, _snapshot_locations_map, _snapshot_prev_key_map

    if row.get('total_tasks_removed', 0) <= 0:
        return []

    # Rebuild cache if tasks DataFrame changed
    tasks_ref = (id(tasks), tasks.shape)
    if _cached_tasks_ref != tasks_ref or _snapshot_locations_map is None or _snapshot_prev_key_map is None:
        grp = tasks.groupby(['id', 'route_id', 'date', 'time'])['location_id'].apply(
            lambda s: [v for v in s.tolist() if pd.notna(v)]
        )
        _snapshot_locations_map = grp.to_dict()

        snaps = tasks[['id', 'route_id', 'date', 'time']].drop_duplicates()
        _snapshot_prev_key_map = {}
        for (route, date), group in snaps.groupby(['route_id', 'date']):
            group_sorted = group.sort_values('time')
            ids = group_sorted['id'].tolist()
            times = group_sorted['time'].tolist()
            for i, row_id in enumerate(ids):
                if i-1 >= 0:
                    prev_id = ids[i-1]
                    prev_time = times[i-1]
                    prev_key = (prev_id, route, date, prev_time)
                else:
                    prev_key = None
                key = (row_id, route, date, times[i])
                _snapshot_prev_key_map[key] = prev_key

        _cached_tasks_ref = tasks_ref

    key = (row.get('id'), row.get('route_id'), row.get('date'), row.get('time'))
    prev_key = _snapshot_prev_key_map.get(key)
    if not prev_key:
        return []

    locations_prev = _snapshot_locations_map.get(prev_key, [])
    locations_current = _snapshot_locations_map.get(key, [])

    # compute locations present in previous snapshot but not in current
    # Preserve the order from the previous snapshot and include multiplicity
    current_counts = Counter(locations_current)
    removed_locations = []
    for loc in locations_prev:
        if current_counts.get(loc, 0) > 0:
            current_counts[loc] -= 1
            continue
        removed_locations.append(loc)
    return removed_locations


def calculate_route_distance(row_id, route_id, date, time, tasks: pd.DataFrame) -> float:
    """
    Calculate route distance in kilometers for a snapshot identified by
    `(row_id, route_id, date, time)` using the Haversine formula.

    Args:
        row_id, route_id, date, time: snapshot key values (matching `tasks` index columns).
        tasks: exploded tasks DataFrame with `latitude`, `longitude`, and `volgorde`.

    Returns:
        Total route distance in kilometers (float).
    """
    # Filter tasks for this snapshot
    mask = (
        (tasks['id'] == row_id) &
        (tasks['route_id'] == route_id) &
        (tasks['date'] == date) &
        (tasks['time'] == time)
    )
    snap = tasks.loc[mask, ['latitude', 'longitude', 'volgorde']].dropna(subset=['latitude', 'longitude'])
    if snap.empty or len(snap) < 2:
        return 0.0

    # Sort by volgorde if present
    if 'volgorde' in snap.columns:
        snap = snap.sort_values('volgorde')

    lat = np.radians(snap['latitude'].to_numpy(dtype=float))
    lon = np.radians(snap['longitude'].to_numpy(dtype=float))

    # Haversine on vector pairs
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R_km = 6371.0
    distances = R_km * c

    return float(np.nansum(distances))


def compute_route_distances_map(tasks: pd.DataFrame) -> dict:
    """
    Compute total route distances (km) for every snapshot present in `tasks`.

    Returns a dict mapping snapshot key tuples `(id, route_id, date, time)` -> distance_km.
    This is much faster than calling `calculate_route_distance` for each row because
    it performs the per-snapshot computation once, iterating groups in Python but
    using vectorized numpy operations inside each group.
    """
    dist_map = {}

    # Only consider rows with valid coordinates
    valid = tasks.dropna(subset=['latitude', 'longitude'])
    if valid.empty:
        return dist_map

    group_cols = ['id', 'route_id', 'date', 'time']
    for key, grp in valid.groupby(group_cols):
        # sort by volgorde if available, else keep current order
        if 'volgorde' in grp.columns:
            grp_sorted = grp.sort_values('volgorde')
        else:
            grp_sorted = grp

        lat = np.radians(grp_sorted['latitude'].to_numpy(dtype=float))
        lon = np.radians(grp_sorted['longitude'].to_numpy(dtype=float))

        if len(lat) < 2:
            dist_map[key] = 0.0
            continue

        dlat = lat[1:] - lat[:-1]
        dlon = lon[1:] - lon[:-1]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R_km = 6371.0
        dist_map[key] = float(np.nansum(R_km * c))

    return dist_map
