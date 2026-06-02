from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from config import (
    HEIGHT_DIR,
    CARBON_DIR,
    REPORT_DIR,
    NODATA,
    AGB_COEF_A,
    AGB_EXP_B,
    CARBON_FRACTION,
    CO2E_FACTOR,
    ensure_directories,
)


def height_to_agb_proxy_ton_per_ha(height_m: np.ndarray) -> np.ndarray:
    """
    Height-derived AGB proxy dalam satuan ton/ha equivalent.

    Catatan:
    - Bukan persamaan biomassa tervalidasi.
    - Tidak memakai DBH, wood density, field plot, atau persamaan alometrik lokal.
    - Dipakai untuk screening relatif, visualisasi, dan portofolio teknis.
    """
    return AGB_COEF_A * np.power(height_m, AGB_EXP_B)


def height_to_carbon_proxy_ton_per_ha(height_m: np.ndarray) -> np.ndarray:
    """
    Height-derived carbon proxy dalam satuan ton C/ha equivalent.
    Penggunaan untuk carbon accounting resmi memerlukan validasi lapangan,
    persamaan biomassa lokal, dan analisis ketidakpastian.
    """
    agb_proxy = height_to_agb_proxy_ton_per_ha(height_m)
    return agb_proxy * CARBON_FRACTION


def generate_carbon_report() -> None:
    ensure_directories()

    height_files = sorted(HEIGHT_DIR.glob("*_canopy_height_proxy.tif"))
    if not height_files:
        raise FileNotFoundError(f"Tidak ada raster canopy height proxy di {HEIGHT_DIR}")

    print("=== Scene-level Carbon Proxy Report ===")

    for height_path in height_files:
        base_name = height_path.name.replace("_canopy_height_proxy.tif", "")

        out_carbon_pixel = CARBON_DIR / f"{base_name}_carbon_proxy_ton_per_pixel_eq.tif"
        out_carbon_ha = CARBON_DIR / f"{base_name}_carbon_proxy_ton_per_ha_eq.tif"
        out_csv = REPORT_DIR / f"{base_name}_scene_report.csv"

        with rasterio.open(height_path) as src:
            height = src.read(1).astype(np.float32)
            src_nodata = src.nodata if src.nodata is not None else NODATA

            px_area_m2 = abs(src.transform.a * src.transform.e)
            px_ha = px_area_m2 / 10000.0

            valid = np.isfinite(height) & (height != src_nodata) & (height > 0)

            if not valid.any():
                print(f"⚠️ Skip {base_name}: tidak ada pixel valid.")
                continue

            agb_proxy_ton_ha = np.zeros_like(height, dtype=np.float32)
            carbon_proxy_ton_ha = np.zeros_like(height, dtype=np.float32)

            agb_proxy_ton_ha[valid] = height_to_agb_proxy_ton_per_ha(
                height[valid]
            ).astype(np.float32)

            carbon_proxy_ton_ha[valid] = height_to_carbon_proxy_ton_per_ha(
                height[valid]
            ).astype(np.float32)

            carbon_proxy_ton_pixel = carbon_proxy_ton_ha * px_ha
            co2e_proxy_ton_pixel = carbon_proxy_ton_pixel * CO2E_FACTOR

            meta = src.meta.copy()
            meta.update(
                dtype="float32",
                nodata=NODATA,
                compress="lzw",
                predictor=2,
            )

            with rasterio.open(out_carbon_pixel, "w", **meta) as dst:
                dst.write(
                    np.where(valid, carbon_proxy_ton_pixel, NODATA).astype(np.float32),
                    1,
                )

            with rasterio.open(out_carbon_ha, "w", **meta) as dst:
                dst.write(
                    np.where(valid, carbon_proxy_ton_ha, NODATA).astype(np.float32),
                    1,
                )

        report = pd.DataFrame(
            {
                "Parameter": [
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
                ],
                "Value": [
                    "HEIGHT_DERIVED_CARBON_PROXY_REQUIRES_FIELD_VALIDATION",
                    float(valid.sum() * px_ha),
                    float(np.mean(height[valid])),
                    float(np.max(height[valid])),
                    float(np.mean(agb_proxy_ton_ha[valid])),
                    float(np.mean(carbon_proxy_ton_ha[valid])),
                    float(np.sum(carbon_proxy_ton_pixel[valid])),
                    float(np.sum(co2e_proxy_ton_pixel[valid])),
                    float(px_ha),
                    f"AGB_proxy_ton_ha_eq = {AGB_COEF_A} * height_m^{AGB_EXP_B}",
                    CARBON_FRACTION,
                    CO2E_FACTOR,
                    (
                        "Height-derived carbon proxy expressed as ton C equivalent for visualization "
                        "and early-stage screening. The values are derived from capped canopy height "
                        "proxy and require field validation, locally calibrated biomass equations, "
                        "and uncertainty assessment before use in official carbon accounting, "
                        "MRV workflows, or carbon credit-related decisions."
                    ),
                ],
            }
        )

        report.to_csv(out_csv, index=False)

        print(f"✅ Carbon proxy ton/pixel raster: {out_carbon_pixel}")
        print(f"✅ Carbon proxy ton/ha raster   : {out_carbon_ha}")
        print(f"✅ Scene report CSV             : {out_csv}")


if __name__ == "__main__":
    generate_carbon_report()