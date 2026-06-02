from pathlib import Path
import argparse
import textwrap

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEIGHT_DIR = PROJECT_ROOT / "data" / "output_maps" / "height"
CARBON_DIR = PROJECT_ROOT / "data" / "output_maps" / "carbon"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"

NODATA_FALLBACK = -9999.0


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def find_first(pattern_dir: Path, pattern: str) -> Path | None:
    files = sorted(pattern_dir.glob(pattern))
    return files[0] if files else None


def read_raster_as_masked(path: Path) -> tuple[np.ma.MaskedArray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK

        mask = (~np.isfinite(arr)) | (arr == nodata)
        masked = np.ma.masked_array(arr, mask=mask)

        meta = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "nodata": nodata,
        }

    return masked, meta


def save_raster_map(
    raster_path: Path,
    out_path: Path,
    title: str,
    colorbar_label: str,
    percentile_clip: bool = True,
) -> None:
    arr, meta = read_raster_as_masked(raster_path)

    valid = arr.compressed()
    if valid.size == 0:
        print(f"⚠️ Skip empty raster: {raster_path}")
        return

    if percentile_clip:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    else:
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(arr, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    subtitle = (
        f"Raster: {raster_path.name}\n"
        f"CRS: {meta['crs']} | Size: {meta['width']} × {meta['height']}"
    )
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=8)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


def save_side_by_side(
    height_path: Path,
    carbon_path: Path,
    out_path: Path,
) -> None:
    height, _ = read_raster_as_masked(height_path)
    carbon, _ = read_raster_as_masked(carbon_path)

    height_valid = height.compressed()
    carbon_valid = carbon.compressed()

    if height_valid.size == 0 or carbon_valid.size == 0:
        print("⚠️ Skip side-by-side: empty raster.")
        return

    h_vmin = float(np.percentile(height_valid, 2))
    h_vmax = float(np.percentile(height_valid, 98))
    c_vmin = float(np.percentile(carbon_valid, 2))
    c_vmax = float(np.percentile(carbon_valid, 98))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes[0].imshow(height, vmin=h_vmin, vmax=h_vmax)
    axes[0].set_title("Canopy Height, capped at 50 m")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.75)
    cb0.set_label("Height (m)")

    im1 = axes[1].imshow(carbon, vmin=c_vmin, vmax=c_vmax)
    axes[1].set_title("Height-derived Carbon Proxy")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.75)
    cb1.set_label("Carbon proxy (ton C/ha eq.)")

    fig.suptitle("Canopy Height and Carbon Proxy Outputs", fontsize=15)

    note = (
        "Carbon proxy is derived from capped canopy height. "
        "Formal carbon accounting requires field validation, locally calibrated biomass equations, "
        "and uncertainty assessment."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


def save_training_log_chart(training_log_path: Path, out_path: Path) -> None:
    if not training_log_path.exists():
        print(f"⚠️ Training log not found: {training_log_path}")
        return

    df = pd.read_csv(training_log_path)

    if "epoch" not in df.columns:
        print("⚠️ training_log.csv does not have an epoch column.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if "train_rmse_m" in df.columns:
        ax.plot(df["epoch"], df["train_rmse_m"], marker="o", markersize=3, label="Train RMSE")

    if "val_rmse_m" in df.columns and df["val_rmse_m"].notna().any():
        ax.plot(df["epoch"], df["val_rmse_m"], marker="o", markersize=3, label="Validation RMSE")

    ax.set_title("U-Net Training Progress Against GLAD Pseudo-label")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    split_mode = df["split_mode"].iloc[0] if "split_mode" in df.columns else "unknown split"
    note = (
        f"Split mode: {split_mode}. "
        "Metrics are evaluated against GLAD Forest Height 2019 pseudo-labels, "
        "not field-measured canopy height."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


def _format_report_value(value) -> str:
    text = str(value)

    try:
        number = float(text)
        if abs(number) >= 1000:
            return f"{number:,.2f}"
        if abs(number) >= 1:
            return f"{number:.2f}"
        return f"{number:.6f}"
    except ValueError:
        return textwrap.fill(text, width=90)


def save_scene_report_table(report_path: Path, out_path: Path) -> None:
    if not report_path.exists():
        print(f"⚠️ Scene report not found: {report_path}")
        return

    df = pd.read_csv(report_path)

    if "Parameter" not in df.columns or "Value" not in df.columns:
        print("⚠️ Scene report must have Parameter and Value columns.")
        return

    keep_params = [
        "Status",
        "Valid area (ha)",
        "Mean predicted canopy height proxy (m)",
        "Max predicted canopy height proxy (m)",
        "Mean height-derived AGB proxy (ton/ha eq.)",
        "Mean height-derived carbon proxy (ton C/ha eq.)",
        "Total height-derived carbon proxy (ton C eq.)",
        "Total height-derived CO2e proxy (ton CO2e eq.)",
        "Pixel area (ha)",
        "AGB proxy formula",
        "Carbon fraction",
        "CO2e factor",
        "Disclaimer",
    ]

    table_df = df[df["Parameter"].isin(keep_params)].copy()
    if table_df.empty:
        print("⚠️ No matching report parameters found for table generation.")
        return

    table_df["Value"] = table_df["Value"].apply(_format_report_value)

    fig_height = max(10, 0.9 * len(table_df) + 3)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df[["Parameter", "Value"]].values,
        colLabels=["Parameter", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.32, 0.68],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.35")
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_text_props(weight="bold", va="center")
            cell.set_height(0.055)
        else:
            cell.set_text_props(va="center")

    for row_idx, param in enumerate(table_df["Parameter"].tolist(), start=1):
        if param == "Disclaimer":
            table[(row_idx, 0)].set_height(0.22)
            table[(row_idx, 1)].set_height(0.22)
            table[(row_idx, 1)].set_text_props(va="center")

        if param == "AGB proxy formula":
            table[(row_idx, 0)].set_height(0.085)
            table[(row_idx, 1)].set_height(0.085)

        if param == "Status":
            table[(row_idx, 0)].set_height(0.07)
            table[(row_idx, 1)].set_height(0.07)

    ax.set_title(
        "Scene-level Canopy Height and Carbon Proxy Report",
        fontsize=16,
        pad=24,
    )

    fig.text(
        0.5,
        0.025,
        f"Source: {report_path.name}",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


def save_output_inventory(out_path: Path) -> None:
    groups = {
        "Height rasters": sorted(HEIGHT_DIR.glob("*.tif")),
        "Carbon rasters": sorted(CARBON_DIR.glob("*.tif")),
        "Reports": sorted(REPORT_DIR.glob("*.csv")),
    }

    lines = []
    for title, files in groups.items():
        lines.append(title)
        if not files:
            lines.append("  - No files found")
        else:
            for file in files:
                rel = file.relative_to(PROJECT_ROOT)
                lines.append(f"  - {rel}")
        lines.append("")

    text = "\n".join(lines)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title("Generated Outputs", fontsize=14, pad=16)
    ax.text(
        0.02,
        0.95,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax.transAxes,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene basename, example: 20220608_Kutai. If omitted, the first available file is used.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.scene:
        height_path = HEIGHT_DIR / f"{args.scene}_canopy_height_proxy.tif"
        carbon_ha_path = CARBON_DIR / f"{args.scene}_carbon_proxy_ton_per_ha_eq.tif"
        scene_report_path = REPORT_DIR / f"{args.scene}_scene_report.csv"
    else:
        height_path = find_first(HEIGHT_DIR, "*_canopy_height_proxy.tif")
        carbon_ha_path = find_first(CARBON_DIR, "*_carbon_proxy_ton_per_ha_eq.tif")
        scene_report_path = find_first(REPORT_DIR, "*_scene_report.csv")

    training_log_path = REPORT_DIR / "training_log.csv"

    if height_path is None or not height_path.exists():
        print(f"⚠️ Height raster not found in {HEIGHT_DIR}")
    else:
        save_raster_map(
            raster_path=height_path,
            out_path=FIGURE_DIR / "03_canopy_height_proxy_map.png",
            title="Canopy Height, capped at 50 m",
            colorbar_label="Height (m)",
        )

    if carbon_ha_path is None or not carbon_ha_path.exists():
        print(f"⚠️ Carbon proxy raster not found in {CARBON_DIR}")
    else:
        save_raster_map(
            raster_path=carbon_ha_path,
            out_path=FIGURE_DIR / "04_carbon_proxy_ton_ha_map.png",
            title="Height-derived Carbon Proxy",
            colorbar_label="Carbon proxy (ton C/ha eq.)",
        )

    if (
        height_path is not None
        and carbon_ha_path is not None
        and height_path.exists()
        and carbon_ha_path.exists()
    ):
        save_side_by_side(
            height_path=height_path,
            carbon_path=carbon_ha_path,
            out_path=FIGURE_DIR / "05_height_carbon_side_by_side.png",
        )

    save_training_log_chart(
        training_log_path=training_log_path,
        out_path=FIGURE_DIR / "06_training_log_rmse.png",
    )

    if scene_report_path is None or not scene_report_path.exists():
        print(f"⚠️ Scene report not found in {REPORT_DIR}")
    else:
        save_scene_report_table(
            report_path=scene_report_path,
            out_path=FIGURE_DIR / "07_scene_report_table.png",
        )

    save_output_inventory(
        out_path=FIGURE_DIR / "08_generated_outputs_inventory.png",
    )

    print("\n🎉 Figures generated.")
    print(f"Output folder: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
