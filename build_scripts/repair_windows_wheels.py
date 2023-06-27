from delvewheel._wheel_repair import WheelRepair
import sys
import pathlib

if __name__ == "__main__":

    args = sys.argv
    wheel_dir = args[1] if len(args) > 1 else "."
    wheel_dir = pathlib.Path(wheel_dir).absolute()
    print(f"Repairing wheels in {wheel_dir}")

    wheels = list(wheel_dir.glob("*.whl"))
    print(f"Found {len(wheels)} wheels")
    
    target_dir = pathlib.Path("./wheelhouse").absolute()
    for wheel in wheels:
        print(f"Repairing {wheel}")
        wr = WheelRepair(whl_path=wheel,
                         extract_dir=None,
                         add_dlls=None,
                         no_dlls=None,
                         ignore_in_wheel=True, 
                         verbose=0,
                         test=[])
        wr.show()
        wr.repair(target=target_dir,
                  no_mangles=set(),
                  no_mangle_all=False,
                  strip=False,
                  lib_sdir=".libs",
                  no_diagnostic=False)
                    