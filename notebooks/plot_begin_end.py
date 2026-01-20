#20/01/2026 00:00

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from time import time
#start1 = time()

EARTH_RADIUS_KM = 6371.0088

# ----------------------------
# Helpers (simple + readable)
# ----------------------------
def safe_json_load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))

def parse_hhmmss_from_filename(name: str) -> str | None:
    parts = name.split("-")
    return parts[2] if len(parts) >= 3 else None

def hhmmss_to_colon(t6: str) -> str:
    if not t6 or len(t6) != 6:
        return ""
    return f"{t6[:2]}:{t6[2:4]}:{t6[4:6]}"

def list_day_files_strict(data_dir: str, subdir: str, route_prefix: str, date_str: str, ext: str) -> list[Path]:
    day_folder = Path(data_dir) / subdir / f"{route_prefix}-{date_str}"
    if not day_folder.exists():
        raise FileNotFoundError(f"Missing day folder: {day_folder}")

    files = sorted(day_folder.glob(f"*.{ext}"), key=lambda p: parse_hhmmss_from_filename(p.name) or "999999")
    if not files:
        raise FileNotFoundError(f"No *.{ext} files found in: {day_folder}")
    return files

def parse_request_tasks_in_order(request_file: Path) -> tuple[dict[str, tuple[float, float]], list[str], str]:
    obj = safe_json_load(request_file)
    config_name = (obj.get("configurationName") or "") if isinstance(obj, dict) else ""

    tasks_map: dict[str, tuple[float, float]] = {}
    task_ids_in_order: list[str] = []

    tasks = obj.get("tasks", []) if isinstance(obj, dict) else []
    for t in tasks or []:
        tid = str((t or {}).get("id", "")).strip()
        addr = (t or {}).get("address", {}) or {}
        lat = float(addr.get("latitude"))
        lon = float(addr.get("longitude"))
        tasks_map[tid] = (lat, lon)
        task_ids_in_order.append(tid)

    return tasks_map, task_ids_in_order, config_name

def read_response_tokens(response_file: Path) -> list[str]:
    return [
        ln.strip()
        for ln in response_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]

def map_ids_to_coords(tasks_map: dict[str, tuple[float, float]], ids: list[str]) -> tuple[list[tuple[float, float]], int]:
    coords: list[tuple[float, float]] = []
    missing = 0
    for tid in ids:
        if tid in tasks_map:
            coords.append(tasks_map[tid])
        else:
            missing += 1
    return coords, missing

def route_length_km(coords_latlon: list[tuple[float, float]]) -> float:
    if len(coords_latlon) < 2:
        return 0.0

    lat_deg = np.array([c[0] for c in coords_latlon], float)
    lon_deg = np.array([c[1] for c in coords_latlon], float)

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    dlat = np.diff(lat)
    dlon = np.diff(lon)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_KM * np.sum(c))

def pretty_title(p: Path) -> str:
    parts = p.name.split("-")
    route = parts[0]
    date = parts[1]
    t6 = parts[2] if len(parts) >= 3 else ""
    return f"{route}  {date[:4]}-{date[4:6]}-{date[6:8]}  {hhmmss_to_colon(t6)}"

def node_key(lat: float, lon: float, nd: int = 6) -> str:
    return f"{round(float(lat), nd)}|{round(float(lon), nd)}"

def key_map_from_tasks(tasks: dict[str, tuple[float, float]], nd: int = 6) -> dict[str, tuple[float, float]]:
    km = {}
    for (lat, lon) in tasks.values():
        km[node_key(lat, lon, nd=nd)] = (lat, lon)
    return km

# ----------------------------
# End selection logic
# ----------------------------
def choose_end_response(resp_files: list[Path], req_by_stem: dict[str, Path], end_choice: str, time_limit: str | None = None) -> Path:
    end_choice = (end_choice or "").strip().lower()

    if end_choice == "last":
        return resp_files[-1]

    if end_choice == "second_last":
        if len(resp_files) < 2:
            raise FileNotFoundError("Need at least 2 response files to select 'second_last'.")
        return resp_files[-2]

    if end_choice == "before_time":
        if not time_limit:
            raise ValueError("end_choice='before_time' requires time_limit='HH:MM:SS'.")
        tl = time_limit.replace(":", "")
        if len(tl) != 6 or not tl.isdigit():
            raise ValueError("time_limit must be in 'HH:MM:SS' format.")
        eligible = []
        for p in resp_files:
            t6 = parse_hhmmss_from_filename(p.name) or ""
            if t6 and t6 <= tl:
                eligible.append(p)
        if not eligible:
            raise FileNotFoundError(f"No response files found with time <= {time_limit}.")
        return eligible[-1]

    if end_choice == "last_non_add":
        for resp_path in reversed(resp_files):
            req_path = req_by_stem.get(resp_path.stem)
            if not req_path:
                continue
            _, _, cfg = parse_request_tasks_in_order(req_path)
            if (cfg or "") != "AddToSequence":
                return resp_path
        raise FileNotFoundError("No matching (response, request) pair found with configurationName != 'AddToSequence'.")

    raise ValueError("end_choice must be one of: 'before_time', 'second_last', 'last', 'last_non_add'.")

def pick_end_by_index(resp_files: list[Path], k: int) -> tuple[Path, int]:
    n = len(resp_files)
    if k <= 0:
        raise ValueError("Internal: pick_end_by_index called with k<=0.")
    idx = (k - 1) if (k - 1) < n else (n - 1)
    return resp_files[idx], (idx + 1)  # chosen 1-based index (after clamping)

# ----------------------------
# Main: BEGIN vs END plots
# ----------------------------
def visualize_begin_end(
    data_dir: str,
    route_prefix: str,
    date_str: str,
    end_choice: str = "second_last",
    time_limit: str | None = None,
    end_pick_no: int = 0,      # 0 neutralised; 1 -> first response; >len -> last response
    nd_round: int = 6,         # rounding for new/deleted node compare
):
    req_files = list_day_files_strict(data_dir, "requests", route_prefix, date_str, "json")
    resp_files = list_day_files_strict(data_dir, "responses", route_prefix, date_str, "txt")
    req_by_stem = {p.stem: p for p in req_files}

    # END response selection (override if end_pick_no > 0)
    if int(end_pick_no) > 0:
        end_resp, chosen_k = pick_end_by_index(resp_files, int(end_pick_no))
        chosen_label = str(chosen_k)  # print number (1-based, clamped)
        use_pick = True
    else:
        end_resp = choose_end_response(resp_files, req_by_stem, end_choice=end_choice, time_limit=time_limit)
        chosen_label = (end_choice or "").strip().upper()
        use_pick = False

    # First output line: chosen end_choice in uppercase OR number
    print(f"Choice: {chosen_label}")

    # BEGIN = first request (order from request)
    begin_req = req_files[0]
    begin_tasks, begin_order_ids, begin_cfg = parse_request_tasks_in_order(begin_req)
    begin_route_latlon, begin_missing = map_ids_to_coords(begin_tasks, begin_order_ids)

    begin_all_lat = np.array([v[0] for v in begin_tasks.values()], float) if begin_tasks else np.array([], float)
    begin_all_lon = np.array([v[1] for v in begin_tasks.values()], float) if begin_tasks else np.array([], float)
    begin_route_lat = np.array([c[0] for c in begin_route_latlon], float) if begin_route_latlon else np.array([], float)
    begin_route_lon = np.array([c[1] for c in begin_route_latlon], float) if begin_route_latlon else np.array([], float)

    # END = selected response, paired request by stem
    end_req = req_by_stem.get(end_resp.stem)
    if not end_req:
        raise FileNotFoundError(f"No matching request JSON for response: {end_resp.name}")

    end_tasks, _, end_cfg = parse_request_tasks_in_order(end_req)
    end_route_ids = read_response_tokens(end_resp)
    end_route_latlon, end_missing = map_ids_to_coords(end_tasks, end_route_ids)

    end_all_lat = np.array([v[0] for v in end_tasks.values()], float) if end_tasks else np.array([], float)
    end_all_lon = np.array([v[1] for v in end_tasks.values()], float) if end_tasks else np.array([], float)
    end_route_lat = np.array([c[0] for c in end_route_latlon], float) if end_route_latlon else np.array([], float)
    end_route_lon = np.array([c[1] for c in end_route_latlon], float) if end_route_latlon else np.array([], float)

    # New/deleted nodes (compare task-location sets by rounded coordinate key)
    km_begin = key_map_from_tasks(begin_tasks, nd=nd_round)
    km_end = key_map_from_tasks(end_tasks, nd=nd_round)

    begin_keys = set(km_begin.keys())
    end_keys = set(km_end.keys())

    new_keys = end_keys - begin_keys
    deleted_keys = begin_keys - end_keys

    new_lat = np.array([km_end[k][0] for k in new_keys], float) if new_keys else np.array([], float)
    new_lon = np.array([km_end[k][1] for k in new_keys], float) if new_keys else np.array([], float)

    del_lat = np.array([km_begin[k][0] for k in deleted_keys], float) if deleted_keys else np.array([], float)
    del_lon = np.array([km_begin[k][1] for k in deleted_keys], float) if deleted_keys else np.array([], float)

    # Stable limits across both plots
    all_lon = np.concatenate([begin_all_lon, end_all_lon]) if len(begin_all_lon) and len(end_all_lon) else (begin_all_lon if len(begin_all_lon) else end_all_lon)
    all_lat = np.concatenate([begin_all_lat, end_all_lat]) if len(begin_all_lat) and len(end_all_lat) else (begin_all_lat if len(begin_all_lat) else end_all_lat)
    if not len(all_lon) or not len(all_lat):
        raise ValueError("No coordinates found to plot (check request JSON content).")

    x_min, x_max = float(np.min(all_lon)), float(np.max(all_lon))
    y_min, y_max = float(np.min(all_lat)), float(np.max(all_lat))
    pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 0.01
    pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 0.01

    fig = plt.figure(figsize=(18, 7))

    # -------------------- Plot 1: BEGIN --------------------
    ax0 = plt.subplot(1, 2, 1)
    ax0.set_title(f"BEGIN (first request)\n{pretty_title(begin_req)}")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_xlim(x_min - pad_x, x_max + pad_x)
    ax0.set_ylim(y_min - pad_y, y_max + pad_y)
    ax0.grid(True)

    # dots black
    ax0.scatter(begin_all_lon, begin_all_lat, s=12, color="black")
    if len(begin_route_lon) >= 2:
        ax0.plot(begin_route_lon, begin_route_lat, linewidth=1.0, alpha=0.70)

    ax0.text(
        0.01, 0.99,
        #f"req: {begin_req.name}\n"
        #f"configurationName: {begin_cfg or '(missing)'}\n"
        #f"tasks: {len(begin_all_lon)}   route_nodes(order from request): {len(begin_route_lon)}\n"
        #f"missing_mapped: {begin_missing}   route: {route_length_km(begin_route_latlon):.3f} km",
        f"route: {route_length_km(begin_route_latlon):.3f} km",
        transform=ax0.transAxes, va="top"
    )

    # -------------------- Plot 2: END --------------------
    ax1 = plt.subplot(1, 2, 2)
    ax1.set_title(f"END (from selected response)\n{pretty_title(end_req)}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(x_min - pad_x, x_max + pad_x)
    ax1.set_ylim(y_min - pad_y, y_max + pad_y)
    ax1.grid(True)

    # dots black
    ax1.scatter(end_all_lon, end_all_lat, s=12, color="black")

    # overlay: new nodes green dots, deleted nodes red X (compared to BEGIN)
    if len(new_lon):
        ax1.scatter(new_lon, new_lat, s=80, color="green")
    if len(del_lon):
        ax1.scatter(del_lon, del_lat, s=80, marker="x", color="red")

    if len(end_route_lon) >= 2:
        ax1.plot(end_route_lon, end_route_lat, linewidth=1.0, alpha=0.70)

    if use_pick:
        select_txt = f"end_pick_no: {chosen_label} (1-based, clamped)"
    else:
        select_txt = f"end_choice: {chosen_label}" + (f" (<= {time_limit})" if chosen_label == "BEFORE_TIME" else "")

    ax1.text(
        0.01, 0.99,
        #f"{select_txt}\n"
        #f"resp: {end_resp.name}\n"
        #f"req : {end_req.name}\n"
        f"configurationName: {end_cfg or '(missing)'}\n"
        #f"tasks: {len(end_all_lon)}   route_nodes(order from response): {len(end_route_lon)}\n"
        #f"missing_mapped: {end_missing}   route: {route_length_km(end_route_latlon):.3f} km\n"
        f"route: {route_length_km(end_route_latlon):.3f} km\n"
        f"new_nodes: {len(new_keys)}   deleted_nodes: {len(deleted_keys)}",
        transform=ax1.transAxes, va="top"
    )

    print(f"Number of files: {len(req_files)}")
    print(f"BEGIN request: {begin_req.name}")
    print(f"END response : {end_resp.name}")
    #print(f"END request  : {end_req.name}")

    plt.tight_layout()
    plt.show()


#print(f"Duration:{time() - start1:.3f}")
