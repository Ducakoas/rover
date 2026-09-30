import importlib
import sys
from pathlib import Path

print("Python:", sys.version)
mods = ["pybullet", "numpy", "scipy", "pandas", "matplotlib", "PIL"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "installed")
        print(f"{m:12s}: OK ({ver})")
    except Exception as exc:
        ok = False
        print(f"{m:12s}: MISSING/ERROR -> {exc}")

root = Path(__file__).resolve().parent
required = [root/"rover_nasa.urdf", root/"assets"/"nasa_stl"/"rover_chassis_body.stl", root/"assets"/"nasa_stl"/"rover_chassis_wheel_tread.stl"]
for p in required:
    print(f"asset {p.name:32s}: {'OK' if p.exists() else 'MISSING'}")
    ok &= p.exists()
print("READY" if ok else "NOT READY")
raise SystemExit(0 if ok else 1)
