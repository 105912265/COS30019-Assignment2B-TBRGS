"""
Streamlit GUI for COS30019 Assignment 2B.

All_data.csv already contains correct WGS84 lat/lon for all 40 sites. The site 4266 (AUBURN_RD N of BURWOOD_RD)'s CSV entry has zeroed out
coordinates; its true position is supplied in SCATS_COORD_FIXES below.

Run from the project root with:
    streamlit run gui/app.py
"""

from __future__ import annotations

import ast
import importlib.util
import io
import math
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import folium
from folium.plugins import AntPath, MeasureControl
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[1]
MAIN_FILE = ROOT_DIR / "main.py"

MODEL_OPTIONS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "rnn": "Simple RNN",
}

DEFAULT_ORIGIN = 2000
DEFAULT_DESTINATION = 3002
DEFAULT_MAX_ROUTES = 5
DEFAULT_MODEL = "lstm"

# SCATS coordinate registry from alldata.csv
import pandas as pd

DATA_FILE = ROOT_DIR / "processed" / "all_data.csv"

# True WGS84 fix for sites whose CSV entry has lat=0 or lon=0.
SCATS_COORD_FIXES: dict[int, tuple[float, float]] = {
    4266: (-37.823_5, 145.043_5),   # AUBURN_RD N of BURWOOD_RD, Hawthorn East
}

# Fallback map centre (inner east Melbourne) used only if the CSV is missing.
MELBOURNE_CENTRE = (-37.820, 145.060)

# Module level cache, loaded once per session.
_SCATS_COORDS: dict[int, tuple[float, float]] | None = None


def _load_scats_coords() -> dict[int, tuple[float, float]]:
    """Load and cache (lat, lon) for every SCATS site from all_data.csv."""
    global _SCATS_COORDS
    if _SCATS_COORDS is not None:
        return _SCATS_COORDS

    coords: dict[int, tuple[float, float]] = {}
    try:
        df = pd.read_csv(DATA_FILE, usecols=["scats_id", "latitude", "longitude"])
        for _, row in df.drop_duplicates("scats_id").iterrows():
            sid = int(row["scats_id"])
            lat, lon = float(row["latitude"]), float(row["longitude"])
            # Apply fix for zeroed out entries
            if sid in SCATS_COORD_FIXES:
                coords[sid] = SCATS_COORD_FIXES[sid]
            elif lat != 0.0 and lon != 0.0:
                coords[sid] = (lat, lon)
            else:
                coords[sid] = SCATS_COORD_FIXES.get(sid, MELBOURNE_CENTRE)
    except Exception:
        pass  # Caller falls back to MELBOURNE_CENTRE

    _SCATS_COORDS = coords
    return coords


def site_coords(site_id: int, _graph=None) -> tuple[float, float]:
    """Return the WGS84 (lat, lon) for a SCATS site ID."""
    return _load_scats_coords().get(site_id, MELBOURNE_CENTRE)


# Traffic colour helpers (Google Maps style)
ROUTE_PALETTE = [
    "#1A73E8",  # Google-blue  — route 1
    "#34A853",  # Google-green — route 2
    "#FBBC04",  # Google-amber — route 3
    "#EA4335",  # Google-red   — route 4
    "#9C27B0",  # Purple       — route 5
]

def travel_time_color(minutes_per_segment: float) -> str:
    """Return a traffic-light colour for a segment's speed."""
    if minutes_per_segment < 1.5:
        return "#00C853"   # fast  — green
    if minutes_per_segment < 3.0:
        return "#FFD600"   # medium — amber
    return "#D50000"       # slow   — red


# main.py loader / runner

def load_main_module():
    spec = importlib.util.spec_from_file_location("tbrgs_main", MAIN_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load main.py")
    module = importlib.util.module_from_spec(spec)
    original_sys_path = list(sys.path)
    try:
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        spec.loader.exec_module(module)
    finally:
        sys.path = original_sys_path
    return module


def run_main_py(model_type: str, origin: int, destination: int, max_routes: int) -> dict[str, Any]:
    params = {
        "model_type": model_type,
        "origin": int(origin),
        "destination": int(destination),
        "max_routes": int(max_routes),
    }
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        main_module = load_main_module()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            routes, graph = main_module.main(
                origin=params["origin"],
                destination=params["destination"],
                model_type=params["model_type"],
                k=params["max_routes"],
            )
        return {
            "returncode": 0,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "params": params,
            "routes": routes or [],
            "graph": graph or {},
        }
    except Exception as exc:
        return {
            "returncode": 1,
            "stdout": stdout_buffer.getvalue(),
            "stderr": f"{type(exc).__name__}: {exc}\n" + stderr_buffer.getvalue(),
            "params": params,
            "routes": [],
            "graph": {},
        }


def parse_main_routes(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = stdout.splitlines()
    index = 0
    while index < len(lines):
        path_match = re.match(r"\s*(\d+)\.\s*Path:\s*(.+?)\s*$", lines[index])
        if not path_match:
            index += 1
            continue
        route_no = int(path_match.group(1))
        raw_path = path_match.group(2)
        path_text = raw_path
        try:
            parsed_path = ast.literal_eval(raw_path)
            if isinstance(parsed_path, list):
                path_text = " → ".join(str(node) for node in parsed_path)
        except Exception:
            pass
        travel_time = ""
        if index + 1 < len(lines):
            time_match = re.search(r"Estimated travel time:\s*([0-9.]+)\s*minutes", lines[index + 1])
            if time_match:
                travel_time = float(time_match.group(1))
        rows.append({
            "Route": route_no,
            "Path": path_text,
            "Estimated travel time (min)": travel_time,
        })
        index += 2
    return rows


# OpenStreetMap map renderer

def _edge_weight(graph: dict | None, u: int, v: int) -> float | None:
    """Extract edge weight from the adjacency dict/list graph structure."""
    if not graph or not isinstance(graph, dict):
        return None
    neighbours = graph.get(u)
    if neighbours is None:
        return None
    if isinstance(neighbours, dict):
        return neighbours.get(v)
    if isinstance(neighbours, list):
        for item in neighbours:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == v:
                return item[1]
    return None


def render_osm_map(routes: list, graph: dict | None = None) -> None:
  
    # Draw an interactive Folium/OpenStreetMap map showing all returned routes.

    if not routes:
        st.info("No routes to display on the map.")
        return

    top_path, top_time = routes[0]
    origin_id      = top_path[0]
    destination_id = top_path[-1]

    # Collect all node coordinates
    all_nodes: set[int] = set()
    for path, _ in routes:
        all_nodes.update(path)

    # Add one hop off path neighbours from the graph
    if graph and isinstance(graph, dict):
        for node in list(all_nodes):
            neighbours = graph.get(node)
            if neighbours is None:
                continue
            if isinstance(neighbours, dict):
                all_nodes.update(neighbours.keys())
            elif isinstance(neighbours, list):
                for item in neighbours:
                    if isinstance(item, (list, tuple)) and item:
                        all_nodes.add(item[0])
                    elif isinstance(item, int):
                        all_nodes.add(item)

    coords: dict[int, tuple[float, float]] = {
        node: site_coords(node, graph) for node in all_nodes
    }

    # Map centre and zoom
    path_lats = [coords[n][0] for n in top_path]
    path_lons = [coords[n][1] for n in top_path]
    centre = (sum(path_lats) / len(path_lats), sum(path_lons) / len(path_lons))

    # Dynamic zoom: tighter bounds → higher zoom
    lat_span = max(path_lats) - min(path_lats) + 1e-6
    lon_span = max(path_lons) - min(path_lons) + 1e-6
    span = max(lat_span, lon_span)
    zoom = max(11, min(16, int(13 - math.log2(span / 0.05))))

    fmap = folium.Map(
        location=centre,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    # CartoDB Positron overlay (cleaner basemap than osm)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
        name="CartoDB Positron (light)",
        max_zoom=19,
    ).add_to(fmap)

    # Each route gets its own toggle in the layer control
    display_routes = routes[:3]
    route_layers = []
    for idx, (rpath, rtime) in enumerate(display_routes):
        label = f"Route {idx + 1} ({round(rtime, 1)} min)"
        layer = folium.FeatureGroup(name=label, show=True)
        route_layers.append(layer)

    layer_offpath = folium.FeatureGroup(name="Off path nodes", show=True)
    layer_markers = folium.FeatureGroup(name="Site markers",   show=True)

    # Off path node markers
    on_path_nodes = set(top_path)
    for node, (lat, lon) in coords.items():
        if node in on_path_nodes:
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color="#888780",
            fill=True,
            fill_color="#F1EFE8",
            fill_opacity=0.7,
            weight=1.5,
            tooltip=folium.Tooltip(f"SCATS {node}"),
        ).add_to(layer_offpath)

    # Draw each route onto its own layer
    for idx, (rpath, rtime) in enumerate(display_routes):
        layer = route_layers[idx]
        colour = ROUTE_PALETTE[idx % len(ROUTE_PALETTE)]
        is_top = idx == 0

        if is_top:
            # Top route: per-segment traffic colouring
            cumulative = 0.0
            for i in range(len(rpath) - 1):
                u, v = rpath[i], rpath[i + 1]
                weight = _edge_weight(graph, u, v) or 0.0
                seg_colour = travel_time_color(weight)
                cumulative += weight
                seg_coords = [coords[u], coords[v]]
                popup_html = (
                    f"<b>{u} to {v}</b><br>"
                    f"Segment time: <b>{round(weight, 2)} min</b><br>"
                    f"Cumulative: <b>{round(cumulative, 2)} min</b>"
                )
                folium.PolyLine(
                    locations=seg_coords,
                    color=seg_colour,
                    weight=9,
                    opacity=0.85,
                    tooltip=f"{u} to {v}: {round(weight, 2)} min",
                    popup=folium.Popup(popup_html, max_width=260),
                ).add_to(layer)
            # Animated direction arrow
            AntPath(
                locations=[coords[n] for n in rpath],
                color="#1A73E8",
                weight=4,
                opacity=0.6,
                delay=600,
                dash_array=[15, 30],
                pulse_color="#FFFFFF",
            ).add_to(layer)
        else:
            # Alternate routes: solid semi-transparent polyline
            route_coords = [coords[n] for n in rpath]
            popup_html = (
                f"<b>Route {idx + 1}</b><br>"
                f"Est. travel time: <b>{round(rtime, 1)} min</b><br>"
                f"Nodes: {' to '.join(str(n) for n in rpath)}"
            )
            folium.PolyLine(
                locations=route_coords,
                color=colour,
                weight=5,
                opacity=0.45,
                tooltip=f"Route {idx + 1}: {round(rtime, 1)} min",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(layer)

    # Site markers for nodes on the top path
    for i, node in enumerate(top_path):
        lat, lon = coords[node]
        cumulative_here = sum(
            _edge_weight(graph, top_path[j], top_path[j + 1]) or 0
            for j in range(i)
        )
        if node == origin_id:
            icon = folium.Icon(color="green", icon="play", prefix="fa")
            marker_label = "Origin"
        elif node == destination_id:
            icon = folium.Icon(color="red", icon="flag-checkered", prefix="fa")
            marker_label = "Destination"
        else:
            icon = folium.Icon(color="blue", icon="map-pin", prefix="fa")
            marker_label = f"{round(cumulative_here, 1)} min"

        popup_html = (
            f"<b>SCATS {node}</b><br>"
            f"{marker_label}<br>"
            f"Lat: {lat:.5f}, Lon: {lon:.5f}"
        )
        folium.Marker(
            location=(lat, lon),
            icon=icon,
            tooltip=folium.Tooltip(f"SCATS {node} ({marker_label})"),
            popup=folium.Popup(popup_html, max_width=220),
        ).add_to(layer_markers)

    # Assemble layers: off path first, then routes back-to-front, markers on top
    layer_offpath.add_to(fmap)
    for layer in reversed(route_layers):
        layer.add_to(fmap)
    layer_markers.add_to(fmap)

    # Controls
    folium.LayerControl(collapsed=False).add_to(fmap)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(fmap)

    # Traffic legend
    legend_html = """
    <div style="
        position: fixed; bottom: 40px; right: 10px; z-index: 9999;
        background: white; border-radius: 10px; padding: 12px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25); font-family: 'Helvetica Neue', sans-serif;
        font-size: 12px; min-width: 170px;">
      <div style="font-weight:700; margin-bottom:8px; color:#202124;">🚦 Traffic Speed</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
        <div style="width:28px;height:6px;border-radius:3px;background:#00C853;"></div>
        <span>Fast (&lt; 1.5 min/seg)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
        <div style="width:28px;height:6px;border-radius:3px;background:#FFD600;"></div>
        <span>Moderate (1.5–3 min)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <div style="width:28px;height:6px;border-radius:3px;background:#D50000;"></div>
        <span>Slow (&gt; 3 min/seg)</span>
      </div>
      <div style="border-top:1px solid #e8eaed; padding-top:8px; font-weight:700;
                  margin-bottom:5px; color:#202124;">🗺 Routes</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <div style="width:28px;height:6px;border-radius:3px;background:#1A73E8;"></div>
        <span>Top route (animated)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:28px;height:6px;border-radius:3px;background:#34A853;opacity:0.55"></div>
        <span>Alt. routes</span>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Route summary panel (top left)
    route_rows = "".join(
        f"<tr><td style='padding:3px 8px;font-weight:700;color:{ROUTE_PALETTE[i % len(ROUTE_PALETTE)]}'>"
        f"Route {i+1}</td>"
        f"<td style='padding:3px 8px;'>{round(t, 1)} min</td>"
        f"<td style='padding:3px 8px;color:#5f6368;font-size:11px;'>"
        f"{' → '.join(str(n) for n in p)}</td></tr>"
        for i, (p, t) in enumerate(routes)
    )
    summary_html = f"""
    <div style="
        position: fixed; top: 10px; left: 55px; z-index: 9999;
        background: white; border-radius: 10px; padding: 12px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25); font-family: 'Helvetica Neue', sans-serif;
        font-size: 12px; max-width: 480px; max-height: 160px; overflow-y: auto;">
      <div style="font-weight:700;margin-bottom:6px;color:#202124;">📍 Route Summary</div>
      <table style="border-collapse:collapse;width:100%">{route_rows}</table>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(summary_html))

    # Render in Streamlit
    map_html = fmap._repr_html_()
    components.html(map_html, height=700, scrolling=False)


# Sidebar

def _load_scats_options() -> list[int]:
    """Return sorted list of SCATS site IDs from all_data.csv."""
    try:
        df = pd.read_csv(DATA_FILE, usecols=["scats_id"]).drop_duplicates()
        return sorted(int(x) for x in df["scats_id"].tolist())
    except Exception:
        return [DEFAULT_ORIGIN, DEFAULT_DESTINATION]


def _load_scats_labels() -> dict[int, str]:
    """Return a mapping of SCATS site ID to display label (ID - location)."""
    try:
        meta_df = pd.read_csv(
            DATA_FILE,
            usecols=["scats_id", "location"]
        ).drop_duplicates("scats_id")

        return {
            int(r["scats_id"]): f"{int(r['scats_id'])} - {r['location']}"
            for _, r in meta_df.iterrows()
        }
    except Exception:
        return {}

def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("TBRGS Settings")
    st.sidebar.write("Select route parameters, then apply them to `main.py`.")

    scats_options = _load_scats_options()
    id_to_label = _load_scats_labels()

    default_origin_idx = (
        scats_options.index(DEFAULT_ORIGIN)
        if DEFAULT_ORIGIN in scats_options else 0
    )

    default_destination_idx = (
        scats_options.index(DEFAULT_DESTINATION)
        if DEFAULT_DESTINATION in scats_options else 0
    )

    draft_model = st.sidebar.selectbox(
        "Traffic prediction model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
        format_func=lambda value: MODEL_OPTIONS[value],
        key="draft_model_type",
    )
    draft_origin = st.sidebar.selectbox(
        "Origin SCATS site",
        options=scats_options,
        index=default_origin_idx,
        format_func=lambda sid: id_to_label.get(sid, str(sid)),
        key="draft_origin",
    )

    draft_destination = st.sidebar.selectbox(
        "Destination SCATS site",
        options=scats_options,
        index=default_destination_idx,
        format_func=lambda sid: id_to_label.get(sid, str(sid)),
        key="draft_destination",
    )
    draft_max_routes = st.sidebar.slider(
        "Maximum routes to return", min_value=1, max_value=5,
        value=DEFAULT_MAX_ROUTES, key="draft_max_routes",
    )

    draft_settings = {
        "model_type": draft_model,
        "origin": int(draft_origin),
        "destination": int(draft_destination),
        "max_routes": int(draft_max_routes),
    }

    if "applied_settings" not in st.session_state:
        st.session_state.applied_settings = draft_settings.copy()

    if st.sidebar.button("Apply settings / Run main.py", type="primary", use_container_width=True):
        st.session_state.applied_settings = draft_settings.copy()
        st.session_state.last_result = run_main_py(**st.session_state.applied_settings)

    if draft_settings != st.session_state.applied_settings:
        st.sidebar.warning("Unapplied changes: press Apply settings / Run main.py.")

    st.sidebar.caption(
        "Currently applied: "
        f"{st.session_state.applied_settings['origin']} → "
        f"{st.session_state.applied_settings['destination']}, "
        f"{MODEL_OPTIONS[st.session_state.applied_settings['model_type']]}, "
        f"up to {st.session_state.applied_settings['max_routes']} route(s)."
    )

    return st.session_state.applied_settings


# Route page

def render_route_page(settings: dict[str, Any]) -> None:
    st.subheader("Route Guidance")
    st.write(
        "Parameters are sent to `main.py`; routes are rendered on an interactive "
        "OpenStreetMap (Folium). Click any segment or marker for details."
    )

    if "last_result" not in st.session_state:
        st.info("Press **Apply settings / Run main.py** in the sidebar to calculate routes.")
        return

    result = st.session_state.last_result
    params = result["params"]

    st.caption(
        f"Applied inputs: {params['origin']} → {params['destination']} | "
        f"Model: {MODEL_OPTIONS.get(params['model_type'], params['model_type'].upper())} | "
        f"Maximum requested routes: {params['max_routes']}"
    )

    if result["returncode"] != 0:
        st.error("main.py returned an error.")
        if "Model not found:" in result["stderr"] or "Scaler not found:" in result["stderr"]:
            st.warning(
                "The selected model cannot run because its trained model/scaler file is missing. "
                "Train the selected model first, or choose a model whose files exist."
            )
        if result["stderr"]:
            st.code(result["stderr"], language="text")
        if result["stdout"]:
            st.write("#### Partial output")
            st.code(result["stdout"], language="text")
        return

    # Interactive OSM map
    all_routes = result.get("routes", [])
    # Map shows at most 3 routes (top + up to 2 alternates)
    display_routes = all_routes[:3]

    if display_routes:
        n_displayed = len(display_routes)
        st.write("#### 🗺 Interactive Map  (OpenStreetMap + Folium)")
        render_osm_map(display_routes, result.get("graph"))
        if n_displayed == 1:
            st.info(
                "Only 1 route was found for this origin–destination pair. "
                "The map shows the single available route."
            )
        elif n_displayed == 2:
            st.info(
                "Only 2 routes were found for this origin–destination pair — "
                "both are shown on the map."
            )
        # 3 routes: no notice, expected maximum
    else:
        st.warning("No routes returned by main.py — map not rendered.")

    # Route table
    routes_table = parse_main_routes(result["stdout"])
    if routes_table:
        st.write("#### All routes")
        st.dataframe(routes_table, use_container_width=True, hide_index=True)
        if len(routes_table) < params["max_routes"]:
            st.info(
                f"main.py returned {len(routes_table)} route(s) "
                f"(requested {params['max_routes']}). "
                "The search only found that many valid paths."
            )

    st.write("#### Raw main.py output")
    st.code(result["stdout"] or "<no stdout>", language="text")


# Status page

def render_status_page() -> None:
    st.subheader("Status")
    st.write("Integration status between the GUI and `main.py`.")

    st.write("#### main.py")
    st.table([
        {"Item": "main.py path",  "Value": str(MAIN_FILE.relative_to(ROOT_DIR))},
        {"Item": "main.py found", "Value": "Yes" if MAIN_FILE.exists() else "No"},
        {"Item": "Map library",   "Value": "Folium + OpenStreetMap (open-source)"},
        {"Item": "Tile source",   "Value": "OpenStreetMap / CartoDB Positron"},
        {"Item": "GUI role",      "Value": "Selection, map rendering, display only"},
    ])

    st.write("#### Coordinates")
    st.info(
        "SCATS site coordinates are loaded directly from `processed/all_data.csv`, "
        "which already contains correct WGS84 lat/lon for all 40 sites. "
        "Site 4266 (AUBURN_RD N of BURWOOD_RD) has zeroed out coordinates in the CSV "
        "and is fixed to its true position in `SCATS_COORD_FIXES` in `gui/app.py`."
    )
    coords = _load_scats_coords()
    coord_rows = [
        {"SCATS site": sid, "Location lat": lat, "Location lon": lon,
         "Source": "fix" if sid in SCATS_COORD_FIXES else "CSV"}
        for sid, (lat, lon) in sorted(coords.items())
    ]
    if coord_rows:
        st.dataframe(coord_rows, use_container_width=True, hide_index=True)

    if "last_result" in st.session_state:
        result = st.session_state.last_result
        st.write("#### Last run")
        st.table([
            {"Item": "Return code",     "Value": result["returncode"]},
            {"Item": "Model",           "Value": MODEL_OPTIONS.get(result["params"]["model_type"], result["params"]["model_type"])},
            {"Item": "Origin",          "Value": result["params"]["origin"]},
            {"Item": "Destination",     "Value": result["params"]["destination"]},
            {"Item": "Maximum routes",  "Value": result["params"]["max_routes"]},
        ])
    else:
        st.info("main.py has not been run from the GUI yet.")


# Entry point

def main() -> None:
    st.set_page_config(page_title="COS30019 TBRGS", page_icon="🚦", layout="wide")
    st.title("Traffic-Based Route Guidance System")
    st.caption("COS30019 Assignment 2B — Interactive map powered by OpenStreetMap or CartoDb & Folium")

    if not MAIN_FILE.exists():
        st.error("main.py was not found in the project root.")
        return

    settings = render_sidebar()

    route_tab, status_tab = st.tabs(["Route Guidance", "Status"])
    with route_tab:
        render_route_page(settings)
    with status_tab:
        render_status_page()


if __name__ == "__main__":
    main()
