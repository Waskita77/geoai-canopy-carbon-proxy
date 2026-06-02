import argparse
import subprocess
import sys
from pathlib import Path

from config import MODEL_PATH, STATS_PATH


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed. Pipeline stopped.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Run tiling and training even if model/statistics already exist.",
    )
    parser.add_argument(
        "--holdout-scene",
        type=str,
        default=None,
        help="Scene stem used for holdout validation. Example: scene_2021_01",
    )
    parser.add_argument(
        "--skip-aoi",
        action="store_true",
        help="Skip 07_aoi_statistics.py.",
    )
    parser.add_argument(
        "--aoi",
        type=str,
        default="../data/aoi/parcels.geojson",
        help="Path to AOI GeoJSON. Default: ../data/aoi/parcels.geojson",
    )
    args = parser.parse_args()

    print("=== GeoAI Canopy & Carbon Proxy Mapping Pipeline ===")
    print("Output type: canopy height and height-derived carbon proxy.")

    if not run_step("01_prepare_project.py"):
        return
    if not run_step("02_align_glad_to_planetscope.py"):
        return

    need_training = args.force_train or (not MODEL_PATH.exists()) or (not STATS_PATH.exists())

    if need_training:
        print("\n⚠️ Model/statistics not found or force-train is enabled. Running tiling and training.")
        if not run_step("03_make_training_tiles.py"):
            return

        train_args = []
        if args.holdout_scene:
            train_args.extend(["--holdout-scene", args.holdout_scene])

        if not run_step("04_train_unet.py", train_args):
            return
    else:
        print("\n✅ Model and normalization statistics found. Skipping tiling/training.")

    if not run_step("05_spatial_inference.py"):
        return
    if not run_step("06_scene_carbon_report.py"):
        return

    if not args.skip_aoi:
        aoi_path = (SCRIPT_DIR / args.aoi).resolve()
        if aoi_path.exists():
            if not run_step("07_aoi_statistics.py", ["--aoi", args.aoi]):
                return
        else:
            print(f"\nℹ️ AOI not found at {aoi_path}. Skipping AOI statistics.")
            print("   To enable AOI statistics, place a GeoJSON at data/aoi/parcels.geojson")

    print("\n🎉 Pipeline completed.")
    print("Main outputs are available in data/output_maps and data/reports.")


if __name__ == "__main__":
    main()
