"""在正式推理前显式编译并加载随包提交的 HIP 源码。"""
from maxvit3d_cpp import ensure_libs


if __name__ == "__main__":
    info = ensure_libs()
    print("HIP build/load OK:", info)
