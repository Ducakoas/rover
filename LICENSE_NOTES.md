# Asset and licensing notes

## Python code
The Python research framework in this archive is newly written for this project.

## NASA meshes
The visual rover meshes under `assets/nasa_stl/` were obtained from the user-supplied rover package. That package states they originate from NASA 3D Resources. The accompanying original notice is preserved at `assets/NASA_ASSET_NOTICE_original.txt`.

For publications, acknowledge NASA as the source of the visual mesh material and verify the current NASA Media Usage Guidelines before redistribution or commercial use. Do not imply NASA endorsement.

## MathWorks files deliberately excluded
The user-supplied MathWorks repository contains a license condition stating that the software, modifications and derivatives are licensed solely for use in conjunction with MathWorks products and service offerings. Therefore this Python archive deliberately does **not** redistribute or depend on MathWorks-specific files such as:

- `mars_terrain.stl`
- `mars_rover_chassis_target_path.mat`
- Simulink implementation files

The Python terrain and reference paths are generated independently.

## Research reproducibility
Every terrain realization is controlled by a seed. Record the seed, scenario, controller case and configuration JSON for every figure/table reported in a manuscript.
