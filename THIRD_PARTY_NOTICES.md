# Third-party notices

This repository contains code from other projects. Each keeps its own license, shown below.
The license files listed here are unmodified copies from the vendored folders; this file only indexes them.

**Non-commercial restriction:** `cumesh/third_party/cubvh` is also under the NVIDIA Source Code License
(`cumesh/third_party/cubvh/LICENSE_NVIDIA`, section 3.3), which allows use "for research or evaluation purposes only".
That restriction applies to that folder; the overall effect on a build that links it has not been assessed here.

| Folder | Upstream project | Copyright holder (as written in the license file) | License | License file |
|---|---|---|---|---|
| `trellis/`, `trellis2/` (TRELLIS and TRELLIS.2 pipelines) | TRELLIS, https://github.com/microsoft/TRELLIS ; TRELLIS.2, https://github.com/microsoft/TRELLIS.2 | Microsoft Corporation | MIT | `LICENSE` (root) |
| `cumesh/` | CuMesh, https://github.com/JeffreyXiang/CuMesh | 2025 Jianfeng XIANG | MIT | `cumesh/LICENSE` |
| `flexgemm_source/` | FlexGEMM, https://github.com/JeffreyXiang/FlexGEMM | 2025 Jianfeng Xiang | MIT | `flexgemm_source/LICENSE` |
| `o_voxel/` (O-Voxel, from the TRELLIS.2 release) | https://github.com/microsoft/TRELLIS.2 | UNKNOWN: `o_voxel/` has no license file of its own; the root `LICENSE` (Microsoft MIT) is the only one that could cover it | MIT (assumed from root `LICENSE`) | `LICENSE` (root) |
| `cumesh/third_party/cubvh/` | cubvh, https://github.com/ashawkey/cubvh | 2022 kiui (MIT file); 2022, NVIDIA Corporation & affiliates (NVIDIA file) | MIT, and NVIDIA Source Code License (non-commercial, research or evaluation only) | `cumesh/third_party/cubvh/LICENSE`, `cumesh/third_party/cubvh/LICENSE_NVIDIA` |
| `cumesh/third_party/xatlas/` | xatlas, https://github.com/jpcy/xatlas | 2018-2020 Jonathan Young | MIT | `cumesh/third_party/xatlas/LICENSE` |
| `o_voxel/third_party/eigen/` | Eigen, https://eigen.tuxfamily.org | See the file headers and `COPYING.*` files in the folder | MPL-2.0 (primarily; some files BSD, Apache or MINPACK, see `COPYING.README`) | `o_voxel/third_party/eigen/LICENSE`, `o_voxel/third_party/eigen/COPYING.MPL2`, `COPYING.BSD`, `COPYING.APACHE`, `COPYING.MINPACK` |
| `trellis/representations/mesh/flexicubes/` | FlexiCubes, https://github.com/nv-tlabs/FlexiCubes (URL not stated in the folder; check) | 2025 NVIDIA CORPORATION & AFFILIATES (from the source header in `flexicubes.py`) | Apache-2.0 | `trellis/representations/mesh/flexicubes/LICENSE.txt` |

## This project's own code

<!-- OWNER: the license for code written for Genesis Forge itself is the owner's decision.
The root `LICENSE` is currently Microsoft's TRELLIS MIT text, and `README.md` says "licensed under MIT". -->

## Not covered

Model weights (TRELLIS, TRELLIS.2, DPT, and others) are downloaded at runtime and carry their own licenses and terms;
they are not vendored here and are not listed above.
