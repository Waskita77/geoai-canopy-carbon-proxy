from config import ensure_directories, GLAD_FILE, RAW_TRAIN_DIR, RAW_INFERENCE_DIR, EXPECTED_BANDS
from utils_raster import list_geotiffs, is_probably_input_raster, validate_multiband_raster


def main() -> None:
    ensure_directories()

    print("=== Prepare Project: GeoAI Canopy & Carbon Proxy Phase 1 ===")
    print("Status klaim: prototype indikatif, bukan MRV resmi.")

    if not GLAD_FILE.exists():
        print(f"⚠️ GLAD belum ditemukan: {GLAD_FILE}")
        print("   Taruh Forest_height_2019_SASIA.tif di data/GLAD/")
    else:
        print(f"✅ GLAD ditemukan: {GLAD_FILE}")

    train_files = [p for p in list_geotiffs(RAW_TRAIN_DIR) if is_probably_input_raster(p)]
    infer_files = [p for p in list_geotiffs(RAW_INFERENCE_DIR) if is_probably_input_raster(p)]

    print(f"Training raster ditemukan : {len(train_files)}")
    print(f"Inference raster ditemukan: {len(infer_files)}")

    for p in train_files + infer_files:
        validate_multiband_raster(p, EXPECTED_BANDS)

    print("✅ Struktur folder dan raster input siap.")


if __name__ == "__main__":
    main()
