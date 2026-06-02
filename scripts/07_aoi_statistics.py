import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from config import (
    HEIGHT_DIR,
    CARBON_DIR,
    REPORT_DIR,
    NODATA,
    CO2E_FACTOR,
    ensure_directories,
)


def _guess_name(row, idx: int) -> str:
    for col in ["name", "Name", "nama", "Nama", "id", "ID"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return f"aoi_{idx + 1}"


def _masked_values(raster_path: Path, geometry, dst_crs) -> tuple[np.ndarray, float]:
    with rasterio.open(raster_path) as src:
        geom_series = gpd.GeoSeries([geometry], crs=dst_crs)

        if geom_series.crs != src.crs:
            geom_series = geom_series.to_crs(src.crs)

        out_img, _ = mask(
            src,
            geom_series.geometry,
            crop=True,
            nodata=src.nodata if src.nodata is not None else NODATA,
            filled=True,
        )

        arr = out_img[0].astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA
        valid = np.isfinite(arr) & (arr != nodata)

        px_area_m2 = abs(src.transform.a * src.transform.e)
        px_ha = px_area_m2 / 10000.0

        return arr[valid], px_ha


def calculate_aoi_statistics(aoi_path: Path) -> None:
    ensure_directories()

    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON tidak ditemukan: {aoi_path}")

    gdf = gpd.read_file(aoi_path)

    if gdf.empty:
        raise ValueError("AOI GeoJSON kosong.")

    if gdf.crs is None:
        raise ValueError(
            "AOI GeoJSON tidak punya CRS. Definisikan CRS dulu, misalnya EPSG:4326."
        )

    height_files = sorted(HEIGHT_DIR.glob("*_canopy_height_proxy.tif"))
    if not height_files:
        raise FileNotFoundError(f"Tidak ada raster height proxy di {HEIGHT_DIR}")

    print("=== AOI Statistics ===")
    print(f"AOI: {aoi_path}")

    for height_path in height_files:
        base_name = height_path.name.replace("_canopy_height_proxy.tif", "")
        carbon_pixel_path = CARBON_DIR / f"{base_name}_carbon_proxy_ton_per_pixel_eq.tif"

        if not carbon_pixel_path.exists():
            print(f"⚠️ Skip {base_name}: carbon proxy raster belum ada: {carbon_pixel_path.name}")
            continue

        rows = []

        with rasterio.open(height_path) as height_src:
            raster_crs = height_src.crs

        gdf_raster = gdf.to_crs(raster_crs)

        for idx, row in gdf_raster.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            aoi_name = _guess_name(row, idx)

            try:
                height_vals, px_ha = _masked_values(height_path, geom, raster_crs)
                carbon_pixel_vals, _ = _masked_values(carbon_pixel_path, geom, raster_crs)

            except ValueError:
                rows.append(
                    {
                        "scene": base_name,
                        "aoi_name": aoi_name,
                        "status": "NO_OVERLAP",
                        "valid_area_ha": 0,
                        "mean_height_proxy_m": None,
                        "max_height_proxy_m": None,
                        "total_carbon_proxy_ton_c_eq": None,
                        "total_co2e_proxy_ton_eq": None,
                        "note": "AOI tidak overlap raster atau tidak ada pixel valid.",
                    }
                )
                continue

            valid_height = height_vals[height_vals > 0]
            valid_carbon = carbon_pixel_vals[carbon_pixel_vals > 0]

            if len(valid_height) == 0 or len(valid_carbon) == 0:
                rows.append(
                    {
                        "scene": base_name,
                        "aoi_name": aoi_name,
                        "status": "NO_VALID_PIXEL",
                        "valid_area_ha": 0,
                        "mean_height_proxy_m": None,
                        "max_height_proxy_m": None,
                        "total_carbon_proxy_ton_c_eq": None,
                        "total_co2e_proxy_ton_eq": None,
                        "note": "AOI overlap raster tetapi tidak ada pixel valid/vegetated.",
                    }
                )
                continue

            total_c = float(np.sum(valid_carbon))

            rows.append(
                {
                    "scene": base_name,
                    "aoi_name": aoi_name,
                    "status": "HEIGHT_DERIVED_CARBON_PROXY_REQUIRES_FIELD_VALIDATION",
                    "valid_area_ha": float(len(valid_height) * px_ha),
                    "mean_height_proxy_m": float(np.mean(valid_height)),
                    "max_height_proxy_m": float(np.max(valid_height)),
                    "total_carbon_proxy_ton_c_eq": total_c,
                    "total_co2e_proxy_ton_eq": float(total_c * CO2E_FACTOR),
                    "note": (
                        "Height-derived carbon proxy from capped canopy height raster. "
                        "Requires field validation, locally calibrated biomass equations, "
                        "and uncertainty assessment before use in official carbon accounting, "
                        "MRV workflows, or carbon credit-related decisions."
                    ),
                }
            )

        out_csv = REPORT_DIR / f"{base_name}_aoi_statistics.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        print(f"✅ AOI statistics CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aoi",
        type=str,
        default="../data/aoi/parcels.geojson",
        help="Path ke AOI GeoJSON. Default: ../data/aoi/parcels.geojson",
    )
    args = parser.parse_args()

    calculate_aoi_statistics(Path(args.aoi))


if __name__ == "__main__":
    main()