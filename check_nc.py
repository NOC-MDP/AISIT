"""
Standalone, serial sanity check for the netCDF files feeding arctic_flux_ano.py.

Both `ecco_paths` (used for ds_s_raw) and `ecco_paths2` (used for ds_v) are just
glob.glob(cfg['inference_target'] + "/*.nc") on the same directory — so the
identical "AttributeError: NetCDF: Can't open HDF5 attribute" showing up
through two different open strategies (serial open_mfdataset, and
parallel=True) points at one or more actually-corrupt files rather than a
locking/concurrency race.

This opens each file one at a time, in a single process, and calls exactly
the operation that's failing in your traceback (ds.ncattrs(), plus each
variable's attrs for good measure) so a bad file surfaces here instead of
mid-pipeline.

Usage:
    python check_netcdf_files.py [directory]

Defaults to cfg['inference_target'] from your script if no directory is given.
"""
import sys
import os
import glob
import netCDF4

DEFAULT_DIR = "/work/scratch-pw5/thopri/ECCO"


def check_file(path):
    with netCDF4.Dataset(path) as ds:
        _ = ds.ncattrs()
        for name, var in ds.variables.items():
            _ = var.ncattrs()
            _ = var.shape  # touches the data layout, not just attrs


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    paths = sorted(glob.glob(os.path.join(target_dir, "*.nc")))
    print(f"Checking {len(paths)} files in {target_dir}\n")

    bad = []
    for i, path in enumerate(paths, 1):
        try:
            check_file(path)
        except Exception as e:
            bad.append((path, repr(e)))
            print(f"[BAD]  {path}\n       {e!r}")
        if i % 50 == 0:
            print(f"  ...checked {i}/{len(paths)}")

    print(f"\nDone. {len(bad)} bad file(s) out of {len(paths)}.")
    if bad:
        print("\nBad files:")
        for path, err in bad:
            print(f"  {path}")
        print(
            "\nNext step: either regenerate these files, or exclude them from "
            "the glob in arctic_flux_ano.py (filter ecco_paths/ecco_paths2 to "
            "drop these paths) so one bad file doesn't kill the whole run."
        )
    else:
        print("All files opened cleanly — the issue is likely something else "
              "(e.g. transient filesystem/locking flakiness under concurrent "
              "access rather than file corruption).")


if __name__ == "__main__":
    main()
