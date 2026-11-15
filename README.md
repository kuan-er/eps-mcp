# EPS MCP

使用 MCP 读取 EFORT PaintStudio（EPS）的工程与喷枪设置，运行已登记的 Windows 原生仿真并导出 OpenUSD，或离线重建相对喷涂沉积。

- [源码和接入示例](#5-部署与交付文件)覆盖只读服务、CPU 重建服务和 Windows 原生桥接。
- [CUDA 冻结版 v1](cuda/eps-cuda-frozen-v1.zip)可以独立复跑。RTX 2080 Ti 上，**1,609／1,616 个工况（99.6%）**的最终整数 UV 沉积网格与原生 EPS 逐值一致。差异统计见[验证记录](evidence/eps-cuda-frozen-v1-validation.json)。
- [原生验收](evidence/native-eps-benchmark/mcp-acceptance.json)记录了 EPS 本身执行仿真、导出 OpenUSD 和 MCP 回读。仓库保留了开枪、关枪和重复回放的样例产物。

仓库基于 2026 年 11 月 11 日交付整理。示例配置使用可替换的安装路径；验收报告中本机用户目录已替换，文件哈希和数值结果保持原样。使用原生执行需要自行提供 EPS、Blender 及本地虚拟工程。

## 使用与验证说明

交付包含两个 MCP 服务。`eps` 用来读取 Windows 原生 EPS 的状态和喷枪设置。`eps_reconstruction` 用来生成轨迹、计算 CPU 沉积，也能通过 `eps_native_scene` 后端驱动已登记的 Windows EPS 工程并导出结果。CUDA 冻结版目前通过独立脚本运行。

| 要做什么 | 使用入口 | 运行条件 |
| --- | --- | --- |
| 查看保存的工程参数、运行状态或喷枪窗口 | `eps` | Windows。实时读取需要同一交互会话中的 EPS 窗口 |
| 生成轨迹并计算相对沉积量 | `eps_reconstruction` 的 `cpu_reference` | Python、NumPy、已登记的工件和喷头配置。可离线执行 |
| 加载工程方案、启动仿真、导出喷涂结果 | `eps_reconstruction` 的 `eps_native_scene` | 交互式 Windows、可运行的 EPS、OpenUSD 录制器、Blender 和桥接模块 |
| 复跑原生采样输入与 CUDA 的数值对照 | `eps-cuda-frozen-v1.zip` | Python、NumPy、NVIDIA 驱动和 NVRTC |

## 1 使用原生 EPS 只读服务

| 工具 | 返回内容 |
| --- | --- |
| `eps_status()` | EPS 进程、Windows 会话、可执行文件版本和可见窗口 |
| `eps_disk_project()` | 固定 `.cmaproj` 文件中的喷枪和喷涂设置，以及本次读取的 SHA-256。`active_project_verified=false` 表明磁盘工程与当前 GUI 的对应关系仍待确认 |
| `eps_live_gun()` | 当前 `GunsWindow` 的 Windows UI Automation 控件和值。需要打开喷枪窗口 |
| `eps_window_screenshot()` | 当前喷枪窗口可见区域的 PNG、文件大小和 SHA-256 |

对照磁盘和现场设置时，分别调用 `eps_disk_project()` 和 `eps_live_gun()`，并保留返回的来源字段。出现 `Expected one EPS process` 时检查同一 Windows 会话中是否有且仅有一个 EPS 进程。实时读取出错会直接返回错误。

## 2 使用 CPU 重建服务

以下为 100 mm 方板 demo。将尖括号内的值替换成上一步返回的 ID，依次调用 MCP 工具。

```text
prepare_case(asset_id="demo_plane")
generate_trajectory(case_id=<case_id>, semantic="top", view_axes=[[0,0,1]], spacing_mm=50)
run_simulation(case_id=<case_id>, trajectory_id=<trajectory_id>, backend="cpu_reference")
get_result(case_id=<case_id>, run_id=<run_id>)
```

用相同 `case_id` 和 `trajectory_id` 再运行一次，并传 `gun_mode="off"`，即可取得关枪对照。成功结果返回 `state="completed"` 和 `array_and_artifact_integrity_verified=true`。`EPS_MCP_CASE_ROOT` 下保存工件快照、轨迹、`run.json`、`raw_deposition.npz`、`metrics.json` 和 `face_metrics.npz`。`get_result` 的 `baseline_run_id` 适用于同一工况、后端、评价口径和数组布局兼容的两次完成运行。

创建工况时默认选 `eps_reconstruction_reference_v1`。最大有效喷距为 300 mm，`EllipseAxisA/B` 为 0.95/0.90，`MaxIncidenceDeg` 为 85°，`FluxScale` 为 10.0。默认相机分辨率为 64×64，采样间隔为 50 ms。工况创建后将所用配置保存到 `gun_profile.json`。自定义工件和喷头应在资产目录中登记，再按 `asset_id` 和 `gun_profile_id` 创建新工况。参考配置没有绑定实物喷枪，沉积量单位为 `relative`。

## 3 执行原生工程并导出结果

`prepare_native_scene` 将场景登记里的 `.cmaproj`、`.cmasln`、`.XPL` 和带 UV 的 `.obj` 绑定到工况，并记录文件哈希。`run_simulation` 由独立 Windows EPS worker 加载方案、运行 XPL、录制并保存 OpenUSD。原生喷枪参数来自工程中的 `GunsFeatures`，开关时序由 XPL 决定。

```text
prepare_native_scene(scene_id="native_spray_target")
run_simulation(case_id=<case_id>, trajectory_id=<trajectory_id>, backend="eps_native_scene")
get_result(case_id=<case_id>, run_id=<run_id>)
```

将第一步场景名改为 `native_spray_target_off` 可以运行同动作的关枪对照。原生后端使用 `prepare_native_scene` 返回的 `trajectory_id`。当前场景支持一个本地回环虚拟机器人和一个静止、仅平移的带 UV 三角化工件。仿真前确保隔离 worker 所在的交互会话可用、其他 EPS 实例已退出、OpenUSD 录制器及英文 `Save` 导出流程可用。

成功运行返回 `state="completed"` 和 `effect_status="NATIVE_EXPORT_WRITTEN"`。随后用 `get_result` 检查 `array_and_artifact_integrity_verified=true`。产物包含原生 `.usda`、执行回执及 `raw_deposition.npz`，其中数组是喷涂结束与基线的整数沉积差值，单位为 EPS 相对量。遇到 `blocked` 先看 `backend_reported_reason`。遇到 `interrupted` 且 `effect_status="UNKNOWN"`，先查同一 EPS 进程、桥接锁、导出和运行回执，再决定是否发起下一次运行。

## 4 核对等价性

| 验证范围 | 已有结果 | 证据文件 |
| --- | --- | --- |
| 原生工程执行与导出 | 已通过加载、仿真、OpenUSD 导出与结果回读。512×512 样例关枪总量为 0，两次开枪总量分别为 437,590 和 1,261,281。原生普通回放存在时序差异 | `evidence/native-eps-benchmark/mcp-acceptance.json` |
| CPU 平板回归 | 本轮使用交付源码从新工况重跑。开枪总量 18,491、72 个非零纹素，关枪为零，两次开枪的 16×16 数组逐值一致 | `evidence/handoff-smoke.json` |
| CPU 与原生整场对照 | 三个有沉积的原生回放与同为 512×512 的 CPU 对照有明显差异。实际采样时序、光栅输入及 UV 结果影响对齐 | `evidence/eps-native-demo/comparison.json` |
| CUDA 冻结版与原生采样对照 | RTX 2080 Ti 上独立回放 1,609/1,616 个整数网格逐值一致。7 个工况共 12 个纹素有差异，最大绝对差为 18。用原生光栅结果作为输入时，命中及最终网格均为 1,616/1,616。受控 C30 两个相位各 16/16 步一致，累计网格也一致 | `evidence/eps-cuda-frozen-v1-validation.json` |
| A100 与 2080 Ti | 早期 CUDA demo 的 5 个网格跨卡逐值一致。冻结版使用更新的源码，A100 上的冻结版复验仍待完成 | `evidence/eps-cuda-a100/cuda-linux-report.json`、`evidence/eps-cuda-windows/cuda-windows-report.json` |

逐值一致指同一 UV 网格内每个整数纹素都相等。比较同一工况前，需要对齐工件、变换、UV 映射和行方向、喷头参数、相机、喷涂时序与基线。物理膜厚、喷枪标定及实际机器人轨迹需要单独验收。

### 复跑冻结版 CUDA

解压 `cuda/eps-cuda-frozen-v1.zip`，在解压根目录执行以下命令。若动态库路径无法找到 NVRTC，追加 `--nvrtc-dir <NVRTC库所在目录>`。

```powershell
python -B holdout/evaluate_holdout.py --backend current --implementation cuda --output replay.json
```

该命令直接复跑包内 192 个贴面工况。本轮在 RTX 2080 Ti 上复现了 190 个独立 CUDA 网格逐值一致，详见 `evidence/handoff-cuda-replay.json`。原生和 CUDA 对照逐例使用同一份 `spray_parameters`，其中 `EllipseAxisA/B` 都为 0.95，`FluxScale` 按工况变化。全量 1,616 个工况的汇总和逐例回执见冻结包的 `validation.json` 和 `receipts/`。普通 EPS Play 的连续整场等价性仍待验证。

## 5 部署与交付文件

本机可以直接使用已注册的 `eps` 和 `eps_reconstruction`。仓库汇集交付源码、配置示例、原生桥接模块、CUDA 冻结包、验证证据和 `SHA256SUMS.txt`。原生 EPS 安装、Blender、完整工程和现场资产需由运行机器提供。

| 交付目录 | 内容 |
| --- | --- |
| `eps/` | 只读 MCP 源码、依赖清单及最小工程配置示例 |
| `eps_reconstruction/` | 当前 MCP 入口、CPU 与原生适配代码、喷头配置和 demo 生成脚本 |
| `native-worker/` | Windows 桥接 DLL、源码、场景登记与 `probe-config` 示例 |
| `cuda/` | 独立 CUDA 冻结包 |
| `evidence/` | 验收报告、开关枪示例导出与本次整理的冒烟测试结果 |

### 新机器安装

1. 克隆仓库或解压源码到较短的本地目录，例如 `C:\EPS-MCP`。在解压目录运行下列命令。验收环境使用 Python 3.14。`setup_demo.py` 生成含本机绝对路径的 `demo/catalog.json`，在新目录执行一次即可。

   ```powershell
   python -m venv eps\.venv
   .\eps\.venv\Scripts\python.exe -m pip install -r eps\mcp\requirements.txt
   python -m venv eps_reconstruction\.venv
   .\eps_reconstruction\.venv\Scripts\python.exe -m pip install -r eps_reconstruction\requirements-offline.txt
   .\eps_reconstruction\.venv\Scripts\python.exe eps_reconstruction\setup_demo.py
   ```

2. 复制 `eps/identity-and-environment.example.json` 为 `eps/identity-and-environment.json`，将 `eps.startup_project` 改为要读取的 `.cmaproj` 绝对路径。
3. 在 MCP 客户端按 `mcp-config.example.toml` 添加两个 stdio 服务，并修改绝对路径。`EPS_MCP_ASSET_CATALOG` 指向 demo 或自有资产目录配置。`EPS_MCP_CASE_ROOT` 建议设为较短的可写目录。若 `get_result` 报 `WinError 206`，缩短该路径后重新创建工况。
4. 需要原生执行时，准备隔离 EPS、Blender 和本地虚拟工程。将随包 DLL 和 `probe-config.json` 放入隔离 EPS 的 `PaintStudio/Modules/NativeRecordingProbe/`。根据示例配置 `probe-config.json` 和场景登记文件，使两边的绝对路径、方案、XPL 和工件名称一致。核对模块 SHA-256、导出 mesh prim 和 UV 行方向，再设置 `EPS_MCP_NATIVE_SCENES` 指向场景登记文件。
5. 在 MCP 客户端刷新工具，先调用 `eps_disk_project()` 和 CPU 平板流程，再执行原生开枪、关枪对照。也可通过 `eps_reconstruction/verify_native_stdio.py` 调用已注册服务。`--config` 传入本机 MCP 客户端的 TOML 配置路径，`--scene` 可取 `cpu_demo`、`native_spray_target` 或 `native_spray_target_off`，`--output` 需指向一个尚不存在的 JSONL 文件。

本轮交付复核了双服务启动、磁盘工程哈希、CPU 开关枪与重复结果，以及两个已注册原生场景的输入绑定。原生启动和导出的结果来自已有端到端验收记录。
