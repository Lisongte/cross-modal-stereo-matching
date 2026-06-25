#!/usr/bin/env python3
"""Export MS2 calib.npy to an OpenCV-readable YAML file.

The MS2 calibration file is a pickled NumPy object array containing a dict.
This parser intentionally avoids importing NumPy so it can run in minimal
environments. Translation vectors in the MS2 file are stored in millimeters;
the exported relative transform uses meters.
"""

from __future__ import annotations

import argparse
import math
import pickle
import struct
import sys
import types
from pathlib import Path


class FakeDType:
    def __init__(self, code, align=False, copy=True):
        self.code = code
        self.byteorder = code[0] if code and code[0] in "<>|=" else "|"
        self.name = code[1:] if self.byteorder in "<>|=" else code

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) >= 2:
            self.byteorder = state[1]


class FakeNDArray:
    def __setstate__(self, state):
        version, shape, dtype, is_fortran, raw = state
        self.shape = tuple(shape)
        self.dtype = dtype
        self.is_fortran = bool(is_fortran)
        self.data = _decode_array(raw, dtype, self.shape, self.is_fortran)

    def tolist(self):
        return self.data


def _reconstruct(subtype, shape, dtype):
    return subtype.__new__(subtype)


def _install_fake_numpy_modules():
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = FakeNDArray
    numpy_mod.dtype = FakeDType

    core_mod = types.ModuleType("numpy.core")
    multiarray_mod = types.ModuleType("numpy.core.multiarray")
    multiarray_mod._reconstruct = _reconstruct

    sys.modules.setdefault("numpy", numpy_mod)
    sys.modules.setdefault("numpy.core", core_mod)
    sys.modules.setdefault("numpy.core.multiarray", multiarray_mod)


def _product(shape):
    n = 1
    for v in shape:
        n *= v
    return n


def _reshape(values, shape, is_fortran):
    if shape == ():
        return values[0]
    if len(shape) == 1:
        return list(values)
    if len(shape) != 2:
        raise ValueError(f"unsupported array shape: {shape}")

    rows, cols = shape
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    k = 0
    if is_fortran:
        for c in range(cols):
            for r in range(rows):
                out[r][c] = values[k]
                k += 1
    else:
        for r in range(rows):
            for c in range(cols):
                out[r][c] = values[k]
                k += 1
    return out


def _decode_array(raw, dtype, shape, is_fortran):
    if not isinstance(raw, bytes):
        return raw

    count = _product(shape)
    code = dtype.code.lstrip("<>|=")
    byteorder = "<" if dtype.byteorder in ("<", "|", "=") else ">"

    if code == "f8":
        values = list(struct.unpack(byteorder + "d" * count, raw))
    elif code == "u1":
        values = list(raw)
    else:
        raise ValueError(f"unsupported dtype in calib.npy: {dtype.code}")
    return _reshape(values, shape, is_fortran)


def load_ms2_calib(path: Path):
    _install_fake_numpy_modules()
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a .npy file")
        major, minor = f.read(2)
        if (major, minor) == (1, 0):
            header_len = struct.unpack("<H", f.read(2))[0]
        elif (major, minor) in ((2, 0), (3, 0)):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"unsupported .npy version: {(major, minor)}")
        f.read(header_len)
        top = pickle.load(f)

    if isinstance(top, FakeNDArray) and top.shape == ():
        return top.data[0]
    raise ValueError("calib.npy did not contain a scalar object dictionary")


def mat_t(m):
    return [list(row) for row in zip(*m)]


def mat_mul(a, b):
    return [
        [sum(a[r][k] * b[k][c] for k in range(len(b))) for c in range(len(b[0]))]
        for r in range(len(a))
    ]


def mat_vec_mul(a, v):
    return [sum(a[r][k] * v[k] for k in range(len(v))) for r in range(len(a))]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def flatten(m):
    if isinstance(m[0], list):
        return [x for row in m for x in row]
    return list(m)


def scale_vec(v, scale):
    if isinstance(v[0], list):
        return [row[0] * scale for row in v]
    return [x * scale for x in v]


def write_matrix(f, name, rows, cols, data):
    f.write(f"{name}: !!opencv-matrix\n")
    f.write(f"   rows: {rows}\n")
    f.write(f"   cols: {cols}\n")
    f.write("   dt: d\n")
    joined = ", ".join(f"{x:.17g}" for x in data)
    f.write(f"   data: [ {joined} ]\n")


def export_rgb_left_to_nir_right(calib, output: Path):
    k_rgb_l = calib["K_rgbL"].tolist()
    k_nir_r = calib["K_nirR"].tolist()

    # Official MS2 dataloader treats these translations as millimeters.
    r_nir_to_rgb = calib["R_nir2rgb"].tolist()
    t_nir_to_rgb_m = scale_vec(calib["T_nir2rgb"].tolist(), 0.001)

    r_nir_l_to_nir_r = calib["R_nirR"].tolist()
    t_nir_l_to_nir_r_m = scale_vec(calib["T_nirR"].tolist(), 0.001)

    # X_rgbL = R_nir2rgb * X_nirL + T_nir2rgb
    # X_nirR = R_nirR * X_nirL + T_nirR
    # Therefore X_nirR = R * X_rgbL + T.
    r_rgb_l_to_nir_r = mat_mul(r_nir_l_to_nir_r, mat_t(r_nir_to_rgb))
    t_rgb_l_to_nir_r = vec_sub(
        t_nir_l_to_nir_r_m,
        mat_vec_mul(r_rgb_l_to_nir_r, t_nir_to_rgb_m),
    )

    baseline_m = math.sqrt(sum(x * x for x in t_rgb_l_to_nir_r))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as f:
        f.write("%YAML:1.0\n")
        f.write("# camera1: RGB left, camera2: NIR right\n")
        f.write("# R,T transform points from RGB-left coordinates to NIR-right coordinates.\n")
        f.write("# Translation values are meters.\n")
        write_matrix(f, "K1", 3, 3, flatten(k_rgb_l))
        write_matrix(f, "D1", 1, 5, [0.0, 0.0, 0.0, 0.0, 0.0])
        write_matrix(f, "K2", 3, 3, flatten(k_nir_r))
        write_matrix(f, "D2", 1, 5, [0.0, 0.0, 0.0, 0.0, 0.0])
        write_matrix(f, "R", 3, 3, flatten(r_rgb_l_to_nir_r))
        write_matrix(f, "T", 3, 1, t_rgb_l_to_nir_r)
        write_matrix(f, "R_nir2rgb", 3, 3, flatten(r_nir_to_rgb))
        write_matrix(f, "T_nir2rgb_m", 3, 1, t_nir_to_rgb_m)
        write_matrix(f, "T_nirR_m", 3, 1, t_nir_l_to_nir_r_m)
        f.write(f"baseline_m: {baseline_m:.17g}\n")

    return {
        "K1": k_rgb_l,
        "K2": k_nir_r,
        "R": r_rgb_l_to_nir_r,
        "T": t_rgb_l_to_nir_r,
        "baseline_m": baseline_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cv_pj/data/calib.npy", type=Path)
    parser.add_argument("--output", default="cv_pj/config/rgb_left_nir_right.yml", type=Path)
    args = parser.parse_args()

    calib = load_ms2_calib(args.input)
    exported = export_rgb_left_to_nir_right(calib, args.output)

    print(f"wrote {args.output}")
    print(f"baseline_m: {exported['baseline_m']:.6f}")
    print("T_rgbL_to_nirR_m:", " ".join(f"{x:.6f}" for x in exported["T"]))


if __name__ == "__main__":
    main()
