# ROVER JOURNAL V8 — WINDOWS-STABLE QP + FAST NATIVE OPENGL VIEWER

V8 sửa hai lỗi thực tế từ V7: C3 crash trong `numpy.linalg.eigvalsh` trên Windows/Conda và custom Tk/TinyRenderer viewer bị lag. Xem `V8_WINDOWS_CRASH_AND_PERFORMANCE_FIXES.md`.

Chạy khuyến nghị:

```powershell
& "C:\\ProgramData\\anaconda3\\envs\\rover311\\python.exe" preview_3d.py
& "C:\\ProgramData\\anaconda3\\envs\\rover311\\python.exe" main.py --case C3 --scenario rough
```

Mặc định V8 dùng `--viewer native` (PyBullet OpenGL trực tiếp, GUI debug bị ẩn) và tốc độ tuning `0.22 m/s`. Muốn dùng viewer telemetry Tk cũ thì thêm `--viewer clean`.

---

# ROVER JOURNAL V6 - WINDOWS VIEWER FIX + BINARY ARM MESHES

> V6 giữ physics PyBullet DIRECT nhưng sửa lỗi Tkinter `couldn't recognize image data` bằng Pillow/ImageTk, đồng thời chuyển các STL cánh tay NASA từ ASCII sang binary STL để TinyRenderer đọc ổn định trên Windows. Các fix anti-stall của V5 vẫn được giữ. Xem `V6_FIXES.md`.

Chạy ngay:

```powershell
& "C:\\ProgramData\\anaconda3\\envs\\rover311\\python.exe" preview_3d.py
& "C:\\ProgramData\\anaconda3\\envs\\rover311\\python.exe" main.py --case C3 --scenario nominal
```

Phím trong cửa sổ 3D: `Q/Esc` thoát, `P` pause/resume, `C` đổi camera `CHASE/WIDE/TOP`. GUI được giữ mở sau khi mô phỏng xong để kiểm tra kết quả.

---

# Six-Wheel Rocker-Bogie Rover Research Framework — V6 Journal Package

Đây là bản Python/PyBullet để **mô phỏng 3D + chạy thí nghiệm control + xuất dữ liệu/figure cho bài báo**. Nó lấy **kiến trúc chức năng** từ Mars Rover example của MathWorks (6 bánh chủ động, 4 bánh lái, Pure Pursuit/Ackermann, uneven terrain, arm sample demo), nhưng phần dynamics/control Python là code độc lập.

> **Quan trọng về license:** project **không chứa** `mars_terrain.stl` hay `rover_path*.mat` của MathWorks vì license của repository MathWorks giới hạn derivatives cho việc dùng cùng MathWorks products. Terrain/path trong project này được tạo độc lập và reproducible. Các STL rover được giữ lại vì package gốc ghi rõ chúng có nguồn từ NASA 3D Resources; xem `assets/NASA_ASSET_NOTICE_original.txt` và `LICENSE_NOTES.md`.

## 1. Nghiên cứu đang được implement

### C0 — baseline
- Pure Pursuit path tracking
- symmetric 4-wheel Ackermann steering
- 6 wheel angular-speed references
- wheel PI torque control

### C1 — slip-aware heuristic
- giữ nguyên outer-loop path tracking
- body-speed PI
- phân bổ torque theo slip: bánh slip nhiều nhận trọng số nhỏ hơn

### C2 — C1 + FTDO
- giữ C1
- thêm Fixed-Time Disturbance Observer (FTDO) cho lumped longitudinal disturbance
- observer bù vào tổng torque demand

### C3 — proposed research controller
- Pure Pursuit + Ackermann giữ nguyên để tách contribution longitudinal/traction
- FTDO
- convex **six-wheel QP torque allocation**
- objective cân bằng: tổng traction demand, wheel-speed synchronization, slip penalty và torque smoothness
- traction-aware box constraints từ `mu_i * Fz_i * R`

## 2. Terrain

Terrain được sinh procedural nhưng deterministic theo seed, gồm nhiều spatial frequency + crater + ridge, nên đẹp và tự nhiên hơn terrain sinusoid bản demo.

Các scenario:
- `flat`
- `nominal`
- `rough`
- `slope`
- `low_mu`
- `split_mu`
- `patchy_mu`
- `rock_field`
- `disturbance`

`disturbance` còn inject lực dọc đã biết ground truth để đánh giá FTDO.

## 3. Cài trên máy hiện tại

Bạn đã tạo environment `rover311`, nên cách chắc nhất là dùng thẳng Python này:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" check_install.py
```

Nếu thiếu package:

```powershell
conda install -n rover311 -c conda-forge pybullet numpy scipy pandas matplotlib pillow -y
```

Không cần `conda activate` nếu PowerShell corporate hook bị lỗi.

## 4. Chạy 3D trước

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" preview_3d.py
```

hoặc double-click:

```text
run_preview.bat
```

## 5. Chạy controller

Baseline:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" main.py --case C0 --scenario nominal
```

Proposed C3:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" main.py --case C3 --scenario nominal
```

Viewer mặc định là `clean`. Nếu cần raw PyBullet để debug:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" main.py --viewer bullet --case C3 --scenario nominal
```

Split-friction:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" main.py --case C3 --scenario split_mu
```

Observer test:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" main.py --case C3 --scenario disturbance
```

Arm demo (không dùng trong experiment chính):

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" main.py --case C3 --scenario nominal --arm-demo
```

## 6. Chạy bộ thí nghiệm journal

Mặc định 4 controller × 5 scenario × 3 seed = 60 runs:

```powershell
& "C:\ProgramData\anaconda3\envs\rover311\python.exe" compare.py --seeds 7 11 19 --scenarios nominal rough low_mu split_mu disturbance
```

Kết quả:

```text
results/journal_suite/
├── raw_metrics.csv
├── aggregate_metrics.csv
├── comparison_tracking_rmse.png
├── comparison_slip_rms.png
├── comparison_specific_energy.png
├── comparison_peak_load.png
└── runs/
    ├── C0_nominal_seed7/
    ├── C1_nominal_seed7/
    ├── ...
    └── C3_disturbance_seed19/
```

Mỗi run có:
- `timeseries.csv`
- `metrics.json`
- `run_metadata.json`
- trajectory
- tracking error
- speed
- 6 wheel slips
- 6 wheel torques
- roll/pitch
- FTDO estimate vs injected disturbance
- 6 normal loads
- spatial friction
- energy proxy

## 7. Metrics dùng cho paper

- cross-track RMSE / max
- heading RMSE / max
- speed RMSE
- RMS / peak slip
- cumulative absolute slip
- RMS / max wheel torque
- max torque rate
- peak normal load
- normal-load coefficient of variation
- contact-loss fraction
- max roll / pitch
- FTDO RMSE against **injected** disturbance
- accumulated `|T omega|` energy proxy
- specific energy proxy [J/m]

## 8. File nào làm gì

```text
rover_sim/config.py       toàn bộ parameter + C0-C3/scenario
rover_sim/paths.py        reference path độc lập
rover_sim/terrain.py      procedural Mars-like heightfield + friction map + rocks
rover_sim/controllers.py  Pure Pursuit/Ackermann, C0, C1/C2 heuristic
rover_sim/observer.py     FTDO
rover_sim/allocator.py    C3 convex QP allocator
rover_sim/disturbance.py  disturbance ground truth
rover_sim/rover.py        PyBullet rover I/O, wheel slip, Fz, friction
rover_sim/simulation.py   vòng lặp simulation + ablation logic
rover_sim/logger.py       CSV, metrics và paper figures
main.py                   single run
compare.py                Monte-Carlo/ablation journal suite
sensitivity.py            gain/weight sensitivity
make_latex_table.py       aggregate CSV -> LaTeX table
preview_3d.py             xem 3D không chạy research controller
paper/                    manuscript scaffold + equations + experiment plan
```

## 9. Nền tảng literature của architecture

1. MathWorks Mars Rover example: Pure Pursuit, wheel steering/speed control, rigid uneven terrain and sample-retrieval manipulator.
2. Toupet et al., *Terrain-adaptive wheel speed control on the Curiosity Mars rover: Algorithm and flight results*, Journal of Field Robotics, 2020, DOI 10.1002/rob.21903.
3. Toupet et al., *Traction Control Design and Integration Onboard the Mars Science Laboratory Curiosity Rover*, IEEE Aerospace Conference, 2018, DOI 10.1109/AERO.2018.8396761.
4. Chen et al., *Simultaneous control of trajectory tracking and coordinated allocation of rocker-bogie planetary rovers*, Mechanical Systems and Signal Processing, 151, 107312, 2021, DOI 10.1016/j.ymssp.2020.107312.
5. Polyakov, *Nonlinear Feedback Design for Fixed-Time Stabilization of Linear Control Systems*, IEEE TAC, 2012, DOI 10.1109/TAC.2011.2179869.

Xem BibTeX trong `paper/references.bib`.

## 10. Những claim KHÔNG được viết quá mức

Plant này dùng **rigid PyBullet contact + Coulomb friction**, không phải deformable Mars regolith. Vì vậy viết:

> rigid uneven terrain with spatially varying wheel-terrain friction

không viết:

> high-fidelity Martian regolith terramechanics

FTDO estimate được đánh giá trực tiếp với ground-truth chỉ cho external disturbance được inject. Địa hình/contact vẫn là lumped unmodeled effects và không có ground-truth scalar duy nhất.

## 11. Trước khi lấy kết quả cuối để nộp

1. `python validate_project.py`
2. `python check_install.py`
3. chạy `C0 nominal` và `C3 nominal` bằng GUI để kiểm tra geometry/contact
4. chạy full `compare.py`
5. kiểm tra mọi run có tới gần goal; loại run bị lật chỉ khi có tiêu chí failure định nghĩa trước, không loại bằng cảm tính
6. chạy `sensitivity.py`
7. dùng `make_latex_table.py`
8. điền số thật vào `paper/manuscript_template.tex`

Đừng chỉnh weight chỉ để ép C3 thắng. Nếu C3 trade-off tracking vs energy/slip, báo cáo đúng trade-off đó.
