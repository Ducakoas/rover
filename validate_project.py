"""Static validation that does not require opening the PyBullet GUI."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
errors=[]
for py in ROOT.rglob("*.py"):
    if "__pycache__" in py.parts: continue
    try: ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    except Exception as e: errors.append(f"Python parse: {py}: {e}")

urdf = ROOT/"rover_nasa.urdf"
try:
    tree=ET.parse(urdf)
    for mesh in tree.findall(".//mesh"):
        fn=mesh.attrib.get("filename","")
        if fn and not (ROOT/fn).exists(): errors.append(f"Missing URDF mesh: {fn}")
except Exception as e: errors.append(f"URDF parse: {e}")

if errors:
    print("VALIDATION FAILED")
    print("\n".join(errors))
    raise SystemExit(1)
print("VALIDATION OK")
