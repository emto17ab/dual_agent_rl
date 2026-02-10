import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import shape

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.patheffects as path_effects

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
DEFAULT_WAGE_JSON = "data/manhattan_wage_data.json"


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


def annotate_centroids(ax, centroids, fmt, font_size):
    for x, y, value in centroids:
        text = ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color="black",
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal(),
            ]
        )


def plot_wage_map(
    geojson_features,
    zip_to_region,
    wages,
    output_path,
    column_width_in,
    annotate=True,
    fmt="${:.1f}",
    font_scale=1.0,
):
    fig, ax = plt.subplots(1, 1, figsize=(column_width_in, column_width_in * 1.1))
    patches, colors, centroids, bounds = create_patches_from_geojson(
        geojson_features, wages, zip_to_region
    )

    collection = PatchCollection(patches, cmap="Oranges", edgecolor="black", linewidth=0.5)
    collection.set_array(np.array(colors))
    collection.set_clim(wages.min(), wages.max())
    ax.add_collection(collection)

    pad = 0.001
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if annotate:
        font_size = max(5.0, min(10.0, 7.5 * font_scale))
        annotate_centroids(ax, centroids, fmt, font_size)

    cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, aspect=22)
    cbar.ax.tick_params(labelsize=max(5.0, 6.5 * font_scale))
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_wage_data(wage_path):
    with open(wage_path, "r") as f:
        wage_data = json.load(f)

    wages = np.array(
        [wage_data["average_salaries"][str(i)] for i in range(len(MANHATTAN_ZIP_CODES))]
    )
    return wages


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate average hourly wage heatmap for Manhattan."
    )
    parser.add_argument(
        "--geojson",
        default=DEFAULT_GEOJSON,
        help="Path to NYC ZCTA GeoJSON.",
    )
    parser.add_argument(
        "--wage-json",
        default=DEFAULT_WAGE_JSON,
        help="Path to Manhattan wage JSON.",
    )
    parser.add_argument(
        "--output",
        default="saved_files/figures/manhattan_average_wage_by_region.png",
        help="Output path for the figure.",
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
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

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

    set_plot_style(base_font_pt, effective_scale)

    manhattan_geojson = load_manhattan_geojson(args.geojson)
    _, zip_to_region = get_region_maps()
    wages = load_wage_data(args.wage_json)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_wage_map(
        manhattan_geojson["features"],
        zip_to_region,
        wages,
        output_path=output_path,
        column_width_in=args.column_width_in,
        annotate=not args.no_annotate,
        font_scale=effective_scale,
    )

    print(f"Saved wage heatmap to: {output_path}")


if __name__ == "__main__":
    main()
