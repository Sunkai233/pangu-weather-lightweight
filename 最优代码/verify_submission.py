"""复核提交目录的必要文件、源码语法、权重和二进制边界。"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "宁静致远_说明文档.pdf",
    "README.md",
    "优化说明文档.md",
    "SUBMISSION_MANIFEST.md",
    "inference.py",
    "result.py",
    "maxvit3d_student.py",
    "maxvit3d_cpp.py",
    "build_hip.py",
    "distill_cache.py",
    "distill_truth.py",
    "muon.py",
    "phys_features.py",
    "training/run_stage1_4090.sh",
    "training/run_stage2_5090.sh",
    "training/assets/metadata.json",
    "src/attn_lib.hip",
    "src/gemm_lib_mmac.hip",
    "conf/config.yaml",
    "data/checkpoints/student.pth",
]
FORBIDDEN_SUFFIXES = {".so", ".o", ".obj", ".dll", ".pyc"}


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("缺少必要文件:\n  " + "\n  ".join(missing))

    bad_syntax = []
    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # noqa: BLE001
            bad_syntax.append(f"{path.relative_to(ROOT)}: {exc}")
    if bad_syntax:
        raise SystemExit("Python 源码语法错误:\n  " + "\n  ".join(bad_syntax))

    binaries = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if binaries:
        raise SystemExit("提交包含预编译/缓存文件:\n  " + "\n  ".join(binaries))

    weight = ROOT / "data/checkpoints/student.pth"
    if weight.stat().st_size <= 0:
        raise SystemExit("学生权重为空")

    try:
        import torch
    except ImportError:
        print("[WARN] 当前环境无 torch，跳过权重内容检查；文件级检查已通过")
    else:
        obj = torch.load(weight, map_location="cpu", weights_only=False)
        state = obj.get("model_state_dict", obj)
        cfg = obj.get("cfg", {}) if isinstance(obj, dict) else {}
        if not isinstance(state, dict) or not state:
            raise SystemExit("学生权重不含有效 model_state_dict")
        if bool(cfg.get("residual", False)):
            raise SystemExit("学生权重 cfg.residual 必须为 False")
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise SystemExit(f"权重张量无效: {name}")

    print(f"提交包自检通过：{len(REQUIRED)} 个必要文件，Python 语法正常，无预编译二进制")


if __name__ == "__main__":
    main()
