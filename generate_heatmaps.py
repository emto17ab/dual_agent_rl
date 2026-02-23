import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

MANHATTAN_ZIP_CODES = [
    "10002",
    "10003",
    "10005",
    "10006",
    "10007",
    "10009",
    "10010",
    "10011",
    "10012",
    "10013",
    "10014",
    "10038",
]

DEFAULT_GEOJSON = "data/nyc-zip-code-tabulation-areas-polygons.geojson"
DEFAULT_VIZ_DIR = "saved_files/visualization_data"


def resolve_model_path(model_name, viz_dir):
    model_path = Path(model_name)
    if model_path.exists():
        return model_path

    if model_name.endswith(".pkl"):
        candidate = Path(viz_dir) / model_name
        if candidate.exists():
            return candidate
    else:
        candidate = Path(viz_dir) / f"{model_name}_viz_data.pkl"
        if candidate.exists():
            return candidate

    matches = sorted(Path(viz_dir).glob(f"*{model_name}*viz_data.pkl"))
    if matches:
        match_list = "\n".join(f"- {m.name}" for m in matches)
        raise FileNotFoundError(
            f"Model not found: {model_name}\nDid you mean one of:\n{match_list}"
        )

    raise FileNotFoundError(f"Model not found: {model_name}")


def load_manhattan_geojson(path):
    with open(path, "r") as f:
        nyc_geojson = json.load(f)

    features = [
        feature
        for feature in nyc_geojson["features"]
        if feature["properties"].get("postalCode") in MANHATTAN_ZIP_CODES
    ]

    return {"type": "FeatureCollection", "features": features}


def get_region_maps():
    region_to_zip = {i: zc for i, zc in enumerate(MANHATTAN_ZIP_CODES)}
    zip_to_region = {zc: i for i, zc in enumerate(MANHATTAN_ZIP_CODES)}
    return region_to_zip, zip_to_region


def create_patches_from_geojson(geojson_features, values, zip_to_region):
    patches = []
    colors = []
    centroids = []
    bounds = [np.inf, np.inf, -np.inf, -np.inf]

    for feature in geojson_features:
        zip_code = feature["properties"]["postalCode"]
        region_idx = zip_to_region[zip_code]

        geom = shape(feature["geometry"])
        centroid = geom.centroid
        centroids.append((centroid.x, centroid.y, values[region_idx]))

        minx, miny, maxx, maxy = geom.bounds
        bounds[0] = min(bounds[0], minx)
        bounds[1] = min(bounds[1], miny)
        bounds[2] = max(bounds[2], maxx)
        bounds[3] = max(bounds[3], maxy)

        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]

        if geom_type == "Polygon":
            for polygon_coords in coords:
                poly = Polygon(polygon_coords, closed=True)
                patches.append(poly)
                colors.append(values[region_idx])
        elif geom_type == "MultiPolygon":
            for multi_poly in coords:
                for polygon_coords in multi_poly:
                    poly = Polygon(polygon_coords, closed=True)
                    patches.append(poly)
                    colors.append(values[region_idx])

    return patches, colors, centroids, bounds


def set_plot_style(base_font_pt, font_scale):
    base = base_font_pt
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base + 1.5,
            "axes.labelsize": base + 0.5,
            "xtick.labelsize": base - 0.5,
            "ytick.labelsize": base - 0.5,
            "figure.dpi": 300,
        }
    )


def _make_axes_grid(num_panels, column_width_in):
    if num_panels == 2:
        return 1, 2, (column_width_in * 2.1, column_width_in * 0.95)

    return 1, 1, (column_width_in, column_width_in * 0.9)


def _annotate_centroids(ax, centroids, fmt, font_size, color="black"):
    for x, y, value in centroids:
        text = ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color=color,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal(),
            ]
        )


def plot_pair_map(
    geojson_features,
    zip_to_region,
    values_a,
    values_b,
    titles,
    captions,
    cmap,
    vmin,
    vmax,
    output_path,
    column_width_in,
    annotate=True,
    fmt="{:.2f}",
    diverging=False,
    font_scale=1.0,
    cbar_tick_step=None,
    cbar_tick_format=None,
    cbar_ticks=None,
):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    for idx, (values, title, caption, suffix) in enumerate(
        [
            (values_a, titles[0], captions[0], "agent0"),
            (values_b, titles[1], captions[1], "agent1"),
        ]
    ):
        fig, ax = plt.subplots(1, 1, figsize=(column_width_in, column_width_in * 1.1))
        patches, colors, centroids, bounds = create_patches_from_geojson(
            geojson_features, values, zip_to_region
        )

        collection = PatchCollection(
            patches, cmap=cmap, edgecolor="black", linewidth=0.5
        )
        collection.set_array(np.array(colors))
        if diverging:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            collection.set_norm(norm)
        else:
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)

        pad = 0.001
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
        ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if annotate:
            font_size = max(5.0, min(10.0, 7.5 * font_scale))
            _annotate_centroids(ax, centroids, fmt, font_size)

        cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, aspect=22)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        elif cbar_tick_step is not None:
            ticks = np.arange(vmin, vmax + 0.5 * cbar_tick_step, cbar_tick_step)
            cbar.set_ticks(ticks)
        if cbar_tick_format is not None:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(cbar_tick_format))
        cbar.ax.tick_params(labelsize=max(5.0, 6.5 * font_scale))
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")

        fig.tight_layout()
        out_path = output_path.with_name(
            f"{output_path.stem}_{suffix}{output_path.suffix}"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_single_map(
    geojson_features,
    zip_to_region,
    values,
    cmap,
    vmin,
    vmax,
    output_path,
    column_width_in,
    annotate=True,
    fmt="{:.2f}",
    diverging=False,
    font_scale=1.0,
    cbar_tick_step=None,
    cbar_tick_format=None,
    cbar_ticks=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(column_width_in, column_width_in * 1.1))
    patches, colors, centroids, bounds = create_patches_from_geojson(
        geojson_features, values, zip_to_region
    )

    collection = PatchCollection(patches, cmap=cmap, edgecolor="black", linewidth=0.5)
    collection.set_array(np.array(colors))
    if diverging:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        collection.set_norm(norm)
    else:
        collection.set_clim(vmin, vmax)
    ax.add_collection(collection)

    pad = 0.001
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if annotate:
        font_size = max(5.0, min(10.0, 7.5 * font_scale))
        _annotate_centroids(ax, centroids, fmt, font_size)

    cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, aspect=22)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    elif cbar_tick_step is not None:
        ticks = np.arange(vmin, vmax + 0.5 * cbar_tick_step, cbar_tick_step)
        cbar.set_ticks(ticks)
    if cbar_tick_format is not None:
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(cbar_tick_format))
    cbar.ax.tick_params(labelsize=max(5.0, 6.5 * font_scale))
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_net_flow_map(
    geojson_features,
    zip_to_region,
    values_a,
    values_b,
    titles,
    captions,
    output_path,
    column_width_in,
    annotate=True,
    fmt="{:.0f}",
    font_scale=1.0,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    cbar_tick_format=None,
):
    if vmin is None or vmax is None:
        max_abs = max(
            abs(values_a.min()),
            abs(values_a.max()),
            abs(values_b.min()),
            abs(values_b.max()),
        )
        vmin = -max_abs
        vmax = max_abs
    plot_pair_map(
        geojson_features,
        zip_to_region,
        values_a,
        values_b,
        titles,
        captions,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        output_path=output_path,
        column_width_in=column_width_in,
        annotate=annotate,
        fmt=fmt,
        diverging=True,
        font_scale=font_scale,
        cbar_ticks=cbar_ticks,
        cbar_tick_format=cbar_tick_format,
    )


def generate_heatmaps(
    model_path,
    geojson_path,
    output_dir,
    column_width_in,
    font_scale,
    base_font_pt,
    annotate,
    demand_max,
    demand_step,
    netflow_range,
    netflow_step,
    price_min,
    price_max,
    price_step,
):
    set_plot_style(base_font_pt, font_scale)

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        flow_data = pickle.load(f)

    manhattan_geojson = load_manhattan_geojson(geojson_path)
    region_to_zip, zip_to_region = get_region_maps()

    def get_agent_array(key, agent_id):
        value = flow_data[key]
        if isinstance(value, dict):
            value = value.get(agent_id)
        else:
            value = value[agent_id]

        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                return None
            value = np.asarray(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return value
        return None

    def has_dual_arrays(key):
        if key not in flow_data:
            return False
        return get_agent_array(key, 0) is not None and get_agent_array(key, 1) is not None

    has_pricing = has_dual_arrays("agent_price_scalars")
    has_demand = has_dual_arrays("agent_demand")
    has_reb_flows = has_dual_arrays("agent_reb_flows") and "edges" in flow_data

    if has_pricing:
        avg_price_agent0 = get_agent_array("agent_price_scalars", 0).mean(axis=0) * 2
        avg_price_agent1 = get_agent_array("agent_price_scalars", 1).mean(axis=0) * 2
        price_min = price_min
        price_max = price_max

        plot_pair_map(
            manhattan_geojson["features"],
            zip_to_region,
            avg_price_agent0,
            avg_price_agent1,
            titles=["Operator 0: Average price", "Operator 1: Average price"],
            captions=[
                "(a) Operator 0: Average price per region.",
                "(b) Operator 1: Average price per region.",
            ],
            cmap="Greens",
            vmin=price_min,
            vmax=price_max,
            output_path=output_dir / "manhattan_avg_pricing_comparison.png",
            column_width_in=column_width_in,
            annotate=annotate,
            fmt="{:.2f}",
            font_scale=font_scale,
            cbar_ticks=np.arange(price_min, price_max + 0.5 * price_step, price_step),
            cbar_tick_format=lambda x, _: f"{x:.1f}",
        )

        price_a0 = get_agent_array("agent_price_scalars", 0)
        price_a1 = get_agent_array("agent_price_scalars", 1)
        if price_a0 is None or price_a1 is None:
            print("Skipping timestep pricing: missing agent_price_scalars.")
        elif price_a0.ndim != 2 or price_a1.ndim != 2:
            print("Skipping timestep pricing: unexpected pricing array shape.")
        else:
            t_last = price_a0.shape[0] - 1
            for t_idx, label in [(0, "t0"), (t_last, "tlast")]:
                plot_single_map(
                    manhattan_geojson["features"],
                    zip_to_region,
                    price_a0[t_idx] * 2,
                    cmap="Greens",
                    vmin=price_min,
                    vmax=price_max,
                    output_path=output_dir
                    / f"manhattan_pricing_{label}_agent0.png",
                    column_width_in=column_width_in,
                    annotate=annotate,
                    fmt="{:.2f}",
                    font_scale=font_scale,
                    cbar_ticks=np.arange(
                        price_min, price_max + 0.5 * price_step, price_step
                    ),
                    cbar_tick_format=lambda x, _: f"{x:.1f}",
                )
                plot_single_map(
                    manhattan_geojson["features"],
                    zip_to_region,
                    price_a1[t_idx] * 2,
                    cmap="Greens",
                    vmin=price_min,
                    vmax=price_max,
                    output_path=output_dir
                    / f"manhattan_pricing_{label}_agent1.png",
                    column_width_in=column_width_in,
                    annotate=annotate,
                    fmt="{:.2f}",
                    font_scale=font_scale,
                    cbar_ticks=np.arange(
                        price_min, price_max + 0.5 * price_step, price_step
                    ),
                    cbar_tick_format=lambda x, _: f"{x:.1f}",
                )
    else:
        print("Skipping pricing heatmaps: missing agent_price_scalars.")

    if has_demand:
        total_demand_agent0 = get_agent_array("agent_demand", 0).sum(axis=0)
        total_demand_agent1 = get_agent_array("agent_demand", 1).sum(axis=0)
        demand_min = 0
        demand_max = demand_max

        plot_pair_map(
            manhattan_geojson["features"],
            zip_to_region,
            total_demand_agent0,
            total_demand_agent1,
            titles=["Operator 0: Total demand", "Operator 1: Total demand"],
            captions=[
                "(a) Operator 0: Total demand per region.",
                "(b) Operator 1: Total demand per region.",
            ],
            cmap="Purples",
            vmin=demand_min,
            vmax=demand_max,
            output_path=output_dir / "manhattan_total_demand_comparison.png",
            column_width_in=column_width_in,
            annotate=annotate,
            fmt="{:.0f}",
            font_scale=font_scale,
            cbar_ticks=np.arange(demand_min, demand_max + 0.5 * demand_step, demand_step),
            cbar_tick_format=lambda x, _: f"{x:.0f}",
        )
    else:
        print("Skipping demand heatmaps: missing agent_demand.")

    if has_reb_flows:
        agent0_flows = get_agent_array("agent_reb_flows", 0)
        agent1_flows = get_agent_array("agent_reb_flows", 1)

        agent0_total_flows = agent0_flows.sum(axis=0)
        agent1_total_flows = agent1_flows.sum(axis=0)

        num_regions = len(MANHATTAN_ZIP_CODES)
        od_matrix_agent0 = np.zeros((num_regions, num_regions))
        od_matrix_agent1 = np.zeros((num_regions, num_regions))

        for edge_idx, (origin, dest) in enumerate(flow_data["edges"]):
            od_matrix_agent0[origin, dest] = agent0_total_flows[edge_idx]
            od_matrix_agent1[origin, dest] = agent1_total_flows[edge_idx]

        total_reb_arrivals_agent0 = od_matrix_agent0.sum(axis=0)
        total_reb_arrivals_agent1 = od_matrix_agent1.sum(axis=0)
        reb_min = min(
            total_reb_arrivals_agent0.min(), total_reb_arrivals_agent1.min()
        )
        reb_max = max(
            total_reb_arrivals_agent0.max(), total_reb_arrivals_agent1.max()
        )

        plot_pair_map(
            manhattan_geojson["features"],
            zip_to_region,
            total_reb_arrivals_agent0,
            total_reb_arrivals_agent1,
            titles=["Agent 0: Rebalancing arrivals", "Agent 1: Rebalancing arrivals"],
            captions=[
                "(a) Agent 0: Total rebalancing arrivals.",
                "(b) Agent 1: Total rebalancing arrivals.",
            ],
            cmap="Blues",
            vmin=reb_min,
            vmax=reb_max,
            output_path=output_dir / "manhattan_total_reb_arrivals_comparison.png",
            column_width_in=column_width_in,
            annotate=annotate,
            fmt="{:.0f}",
            font_scale=font_scale,
        )

        total_reb_departures_agent0 = od_matrix_agent0.sum(axis=1)
        total_reb_departures_agent1 = od_matrix_agent1.sum(axis=1)
        dep_min = min(
            total_reb_departures_agent0.min(), total_reb_departures_agent1.min()
        )
        dep_max = max(
            total_reb_departures_agent0.max(), total_reb_departures_agent1.max()
        )

        plot_pair_map(
            manhattan_geojson["features"],
            zip_to_region,
            total_reb_departures_agent0,
            total_reb_departures_agent1,
            titles=[
                "Agent 0: Rebalancing departures",
                "Agent 1: Rebalancing departures",
            ],
            captions=[
                "(a) Agent 0: Total rebalancing departures.",
                "(b) Agent 1: Total rebalancing departures.",
            ],
            cmap="Reds",
            vmin=dep_min,
            vmax=dep_max,
            output_path=output_dir / "manhattan_total_reb_departures_comparison.png",
            column_width_in=column_width_in,
            annotate=annotate,
            fmt="{:.0f}",
            font_scale=font_scale,
        )

        net_reb_flow_agent0 = total_reb_arrivals_agent0 - total_reb_departures_agent0
        net_reb_flow_agent1 = total_reb_arrivals_agent1 - total_reb_departures_agent1

        plot_net_flow_map(
            manhattan_geojson["features"],
            zip_to_region,
            net_reb_flow_agent0,
            net_reb_flow_agent1,
            titles=[
                "Operator 0: Net rebalancing flow",
                "Operator 1: Net rebalancing flow",
            ],
            captions=[
                "(a) Operator 0: Net rebalancing flow.",
                "(b) Operator 1: Net rebalancing flow.",
            ],
            output_path=output_dir / "manhattan_net_reb_flow_comparison.png",
            column_width_in=column_width_in,
            annotate=annotate,
            fmt="{:.0f}",
            font_scale=font_scale,
            vmin=-netflow_range,
            vmax=netflow_range,
            cbar_ticks=np.arange(
                -netflow_range + (netflow_range % netflow_step),
                netflow_range + 0.5 * netflow_step,
                netflow_step,
            ),
            cbar_tick_format=lambda x, _: f"{x:.0f}",
        )
    else:
        print("Skipping rebalancing heatmaps: missing agent_reb_flows or edges.")

    print(f"Saved heatmaps to: {output_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate matplotlib heatmaps for a model visualization pickle."
    )
    parser.add_argument(
        "model_name",
        help=(
            "Model name or path. If a name is provided, the script looks for "
            "saved_files/visualization_data/<name>_viz_data.pkl"
        ),
    )
    parser.add_argument(
        "--geojson",
        default=DEFAULT_GEOJSON,
        help="Path to NYC ZCTA GeoJSON.",
    )
    parser.add_argument(
        "--viz-dir",
        default=DEFAULT_VIZ_DIR,
        help="Directory holding *_viz_data.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--column-width-in",
        type=float,
        default=3.35,
        help="Single-column width in inches.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale factor for fonts and annotations.",
    )
    parser.add_argument(
        "--font-pt",
        type=float,
        default=None,
        help="Base font size in points (overrides --font-scale).",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable numeric annotations inside regions.",
    )
    parser.add_argument(
        "--demand-max",
        type=float,
        default=600.0,
        help="Fixed max value for demand colorbars.",
    )
    parser.add_argument(
        "--demand-step",
        type=float,
        default=100.0,
        help="Tick step for demand colorbars.",
    )
    parser.add_argument(
        "--netflow-range",
        type=float,
        default=90.0,
        help="Absolute range for netflow colorbars.",
    )
    parser.add_argument(
        "--netflow-step",
        type=float,
        default=20.0,
        help="Tick step for netflow colorbars.",
    )
    parser.add_argument(
        "--price-min",
        type=float,
        default=1.1,
        help="Fixed min value for pricing colorbars.",
    )
    parser.add_argument(
        "--price-max",
        type=float,
        default=1.5,
        help="Fixed max value for pricing colorbars.",
    )
    parser.add_argument(
        "--price-step",
        type=float,
        default=0.1,
        help="Tick step for pricing colorbars.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_name, args.viz_dir)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path("saved_files/figures") / model_path.stem

    if args.font_pt is not None:
        base_font_pt = args.font_pt
        effective_scale = base_font_pt / 7.5
    elif args.font_scale > 4:
        base_font_pt = args.font_scale
        effective_scale = base_font_pt / 7.5
        print("Interpreting --font-scale as point size. Use --font-pt to be explicit.")
    else:
        effective_scale = args.font_scale
        base_font_pt = 7.5 * args.font_scale

    generate_heatmaps(
        model_path=model_path,
        geojson_path=args.geojson,
        output_dir=output_dir,
        column_width_in=args.column_width_in,
        font_scale=effective_scale,
        base_font_pt=base_font_pt,
        annotate=not args.no_annotate,
        demand_max=args.demand_max,
        demand_step=args.demand_step,
        netflow_range=args.netflow_range,
        netflow_step=args.netflow_step,
        price_min=args.price_min,
        price_max=args.price_max,
        price_step=args.price_step,
    )


if __name__ == "__main__":
    main()
