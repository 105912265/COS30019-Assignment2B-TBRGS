"""
Streamlit GUI for COS30019 Assignment 2B.

This file is deliberately a thin graphical interface. It does not load the
processed dataset, route graph, model files, configuration file, or project
algorithm modules itself. The selected GUI parameters are passed directly to
main.py, and the printed result from main.py is displayed in the GUI.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

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


def load_main_module():
    """Load main.py so the GUI can call main.main(...) directly."""

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
    """Call main.py with GUI-selected parameters and capture its printed output."""

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
    """Parse the text route output produced by main.py."""

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

        rows.append(
            {
                "Route": route_no,
                "Path": path_text,
                "Estimated travel time (min)": travel_time,
            }
        )
        index += 2

    return rows


def render_path_graph(routes: list, graph: dict | None = None) -> None:
    """
    Draw a graph visualisation of the top path styled like the reference image:
    red filled nodes and edges for the top path, green for the destination,
    open gray circles for off-path nodes, edge weights labelled on every edge.
    """
    if not routes:
        return

    top_path, top_time = routes[0]
    if len(top_path) < 2:
        st.warning("Top path has fewer than 2 nodes — nothing to draw.")
        return

    origin = top_path[0]
    destination = top_path[-1]
    top_edges = set(zip(top_path[:-1], top_path[1:]))

    # Build a NetworkX directed graph
    G = nx.DiGraph()
    for node in top_path:
        G.add_node(node)
    for u, v in top_edges:
        w = None
        if graph and u in graph:
            neighbours = graph[u]
            if isinstance(neighbours, dict) and v in neighbours:
                w = round(neighbours[v], 2)
        G.add_edge(u, v, weight=w)

    # Add one-hop off-path neighbours — handle both dict and list adjacency
    if graph:
        for node in top_path:
            neighbours = graph.get(node) if hasattr(graph, "get") else None
            if neighbours is None:
                continue
            if isinstance(neighbours, dict):
                items = neighbours.items()
            elif isinstance(neighbours, list):
                # List of (neighbour, weight) tuples or just neighbour IDs
                if neighbours and isinstance(neighbours[0], (list, tuple)):
                    items = [(nb, w) for nb, w in neighbours]
                else:
                    items = [(nb, None) for nb in neighbours]
            else:
                continue
            for neighbour, weight in items:
                if neighbour not in G:
                    G.add_node(neighbour)
                if not G.has_edge(node, neighbour):
                    G.add_edge(node, neighbour, weight=round(weight, 2) if weight is not None else None)

    # Layout: top-path nodes evenly left-to-right at y=0,
    # off-path neighbours staggered above/below
    pos = {}
    n = len(top_path)
    for i, node in enumerate(top_path):
        pos[node] = (i * 2.0, 0.0)

    off_path_nodes = [node for node in G.nodes if node not in top_path]
    for i, node in enumerate(off_path_nodes):
        angle = math.pi * (i + 1) / (len(off_path_nodes) + 1)
        pos[node] = (
            (n - 1) * angle / math.pi * 2.0,
            1.6 * (1 if i % 2 == 0 else -1),
        )

    RED_FILL   = "#E24B4A"
    RED_LIGHT  = "#FCEBEB"
    GREEN_FILL = "#EAF3DE"
    GREEN_EDGE = "#3B6D11"
    GRAY_FILL  = "#F1EFE8"
    GRAY_EDGE  = "#888780"
    OFF_EDGE   = "#C8C6BE"

    node_colors, node_edge_colors, node_lws = [], [], []
    for node in G.nodes:
        if node == destination:
            node_colors.append(GREEN_FILL)
            node_edge_colors.append(GREEN_EDGE)
            node_lws.append(2.5)
        elif node in top_path:
            node_colors.append(RED_LIGHT)
            node_edge_colors.append(RED_FILL)
            node_lws.append(2.0)
        else:
            node_colors.append(GRAY_FILL)
            node_edge_colors.append(GRAY_EDGE)
            node_lws.append(1.0)

    top_edge_list = [(u, v) for u, v in G.edges if (u, v) in top_edges]
    off_edge_list = [(u, v) for u, v in G.edges if (u, v) not in top_edges]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    nx.draw_networkx_edges(
        G, pos, edgelist=off_edge_list, ax=ax,
        edge_color=OFF_EDGE, width=1.0, arrows=True,
        arrowstyle="-|>", arrowsize=12,
        connectionstyle="arc3,rad=0.08", node_size=900,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=top_edge_list, ax=ax,
        edge_color=RED_FILL, width=2.5, arrows=True,
        arrowstyle="-|>", arrowsize=16,
        connectionstyle="arc3,rad=0.08", node_size=900,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(G.nodes), ax=ax,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=node_lws,
        node_size=900,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8, font_weight="bold", font_color="#2C2C2A",
    )

    # Cumulative travel-time annotations above each top-path node
    for i, node in enumerate(top_path):
        x, y = pos[node]
        cumulative = sum(
            G[top_path[j]][top_path[j + 1]].get("weight") or 0
            for j in range(i)
        )
        label = "origin" if i == 0 else f"{round(cumulative, 1)} min"
        color = GREEN_EDGE if node == destination else RED_FILL
        ax.annotate(
            label, xy=(x, y), xytext=(x, y + 0.52),
            ha="center", va="bottom", fontsize=7.5,
            color=color, fontweight="bold",
        )

    edge_labels = {
        (u, v): (str(d["weight"]) if d.get("weight") is not None else "")
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=8, font_color="#5F5E5A",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )

    legend_handles = [
        mpatches.Patch(facecolor=RED_LIGHT,  edgecolor=RED_FILL,  linewidth=1.5, label="Top path node"),
        mpatches.Patch(facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, linewidth=1.5, label="Destination"),
        mpatches.Patch(facecolor=GRAY_FILL,  edgecolor=GRAY_EDGE,  linewidth=1.0, label="Off-path node"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Top path: {' → '.join(str(node) for node in top_path)}   |   "
        f"Estimated travel time: {round(top_time, 2)} min",
        fontsize=10, color="#2C2C2A", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


def render_sidebar() -> dict[str, Any]:
    """Render GUI-only settings and return the currently applied values."""

    st.sidebar.header("TBRGS Settings")
    st.sidebar.write("Select route parameters, then apply them to `main.py`.")

    draft_model = st.sidebar.selectbox(
        "Traffic prediction model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
        format_func=lambda value: MODEL_OPTIONS[value],
        key="draft_model_type",
    )

    draft_origin = st.sidebar.number_input(
        "Origin SCATS site",
        min_value=0,
        max_value=999999,
        value=DEFAULT_ORIGIN,
        step=1,
        key="draft_origin",
    )

    draft_destination = st.sidebar.number_input(
        "Destination SCATS site",
        min_value=0,
        max_value=999999,
        value=DEFAULT_DESTINATION,
        step=1,
        key="draft_destination",
    )

    draft_max_routes = st.sidebar.slider(
        "Maximum routes to return",
        min_value=1,
        max_value=5,
        value=DEFAULT_MAX_ROUTES,
        key="draft_max_routes",
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


def render_route_page(settings: dict[str, Any]) -> None:
    """Display the main.py result for the selected route settings."""

    st.subheader("Route Guidance")
    st.write(
        "This GUI page sends the selected sidebar parameters to `main.py` and displays "
        "the output returned by `main.py`. The GUI does not calculate routes itself."
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
                "Train the selected model first, or choose a model whose files exist. "
                "The GUI has passed the selected parameters to main.py correctly; main.py failed "
                "inside the existing prediction pipeline."
            )
        if result["stderr"]:
            st.code(result["stderr"], language="text")
        if result["stdout"]:
            st.write("#### Partial output")
            st.code(result["stdout"], language="text")
        return

    # Graph visualisation of the top path
    if result.get("routes"):
        st.write("#### Top path visualisation")
        render_path_graph(result["routes"], result.get("graph"))

    routes = parse_main_routes(result["stdout"])
    if routes:
        st.write("#### All routes")
        st.dataframe(routes, use_container_width=True, hide_index=True)
        if len(routes) < params["max_routes"]:
            st.info(
                f"main.py returned {len(routes)} route(s), even though {params['max_routes']} "
                "were requested. This means the graph/search implementation behind main.py "
                "only found that many valid routes for the selected origin and destination."
            )
    else:
        st.warning("main.py completed, but no route rows were found in its output.")

    st.write("#### Raw main.py output")
    st.code(result["stdout"] or "<no stdout>", language="text")


def render_status_page() -> None:
    """Display GUI-to-main.py integration status."""

    st.subheader("Status")
    st.write("This page shows whether the GUI can locate and call `main.py`.")

    st.write("#### main.py")
    st.table(
        [
            {
                "Item": "main.py path",
                "Value": str(MAIN_FILE.relative_to(ROOT_DIR)),
            },
            {
                "Item": "main.py found",
                "Value": "Yes" if MAIN_FILE.exists() else "No",
            },
            {
                "Item": "GUI role",
                "Value": "Selection and display only",
            },
        ]
    )

    if "last_result" in st.session_state:
        result = st.session_state.last_result
        st.write("#### Last run")
        st.table(
            [
                {"Item": "Return code", "Value": result["returncode"]},
                {"Item": "Model", "Value": MODEL_OPTIONS.get(result["params"]["model_type"], result["params"]["model_type"])},
                {"Item": "Origin", "Value": result["params"]["origin"]},
                {"Item": "Destination", "Value": result["params"]["destination"]},
                {"Item": "Maximum routes", "Value": result["params"]["max_routes"]},
            ]
        )
    else:
        st.info("main.py has not been run from the GUI yet.")


def main() -> None:
    st.set_page_config(page_title="COS30019 TBRGS", page_icon="🚦", layout="wide")
    st.title("Traffic-Based Route Guidance System")
    st.caption("COS30019 Assignment 2B — GUI wrapper around main.py")

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