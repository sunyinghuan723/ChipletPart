#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate floorplan hotspot with DeepOHeat."
    )
    parser.add_argument("--input", required=True, help="Path to thermal layout JSON.")
    parser.add_argument(
        "--deepoheat-root",
        required=True,
        help="Path to the DeepOHeat repository root.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the DeepOHeat checkpoint.",
    )
    parser.add_argument(
        "--reference-total-power",
        type=float,
        default=0.25,
        help="Normalize all layouts to this total integrated power.",
    )
    return parser.parse_args()


def add_deepoheat_to_path(deepoheat_root: Path):
    if not deepoheat_root.exists():
        raise FileNotFoundError(f"DeepOHeat root not found: {deepoheat_root}")
    sys.path.insert(0, str(deepoheat_root))


def build_model(modules, device: str):
    model = modules.DeepONet(
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=441,
        branch_hidden_features=256,
        inner_prod_features=128,
        num_trunk_hidden_layers=3,
        num_branch_hidden_layers=7,
        nonlinearity="silu",
        freq=2 * torch.pi,
        std=1,
        freq_trainable=True,
        device=device,
    )
    return model


def build_dataset(dataio_deeponet):
    domain_0 = dict(
        domain_name=0,
        geometry=dict(
            starts=[0.0, 0.0, 0.0],
            ends=[1.0, 1.0, 0.5],
            num_intervals=[20, 20, 10],
            num_pde_points=2000,
            num_single_bc_points=200,
        ),
        conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
        power=dict(
            bc=True,
            num_power_points_per_volume=2,
            num_power_points_per_surface=500,
            num_power_points_per_cell=5,
            power_map=dict(
                power_0=dict(
                    type="surface_power",
                    surface="top",
                    location=dict(starts=(10, 0, 10), ends=(20, 10, 10)),
                    params=dict(dim=2, value=1, weight=1),
                )
            ),
        ),
        front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
        back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
        left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
        right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
        bottom=dict(
            bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)
        ),
        top=dict(bc=True),
        node=dict(root=True, leaf=True),
        parameterized=dict(variable=False),
    )
    domains_list = [domain_0]
    global_params = {
        "loss_fun_type": "norm",
        "num_params_per_epoch": 5,
        "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
    }
    return dataio_deeponet.DeepONetMeshDataIO(
        domains_list, global_params, dim=2, var=1, len_scale=0.3
    )


def overlap_length(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def normalize_rectangles(chiplets):
    min_x = min(chiplet["x"] for chiplet in chiplets)
    min_y = min(chiplet["y"] for chiplet in chiplets)
    max_x = max(chiplet["x"] + chiplet["width"] for chiplet in chiplets)
    max_y = max(chiplet["y"] + chiplet["height"] for chiplet in chiplets)

    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    span = max(span_x, span_y)

    pad_x = 0.5 * (span - span_x)
    pad_y = 0.5 * (span - span_y)

    normalized = []
    for chiplet in chiplets:
        normalized.append(
            {
                "power": chiplet["power"],
                "x": (chiplet["x"] - min_x + pad_x) / span,
                "y": (chiplet["y"] - min_y + pad_y) / span,
                "width": chiplet["width"] / span,
                "height": chiplet["height"] / span,
            }
        )
    return normalized


def rasterize_power_map(chiplets, grid_size: int = 20):
    power_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    for chiplet in normalize_rectangles(chiplets):
        area = chiplet["width"] * chiplet["height"]
        if area <= 0.0 or chiplet["power"] <= 0.0:
            continue

        power_density = chiplet["power"] / area
        x0 = chiplet["x"]
        x1 = chiplet["x"] + chiplet["width"]
        y0 = chiplet["y"]
        y1 = chiplet["y"] + chiplet["height"]

        col_start = max(0, min(grid_size - 1, int(math.floor(x0 * grid_size))))
        col_end = max(0, min(grid_size - 1, int(math.ceil(x1 * grid_size)) - 1))
        row_start = max(0, min(grid_size - 1, int(math.floor(y0 * grid_size))))
        row_end = max(0, min(grid_size - 1, int(math.ceil(y1 * grid_size)) - 1))

        for row in range(row_start, row_end + 1):
            cell_y0 = row / grid_size
            cell_y1 = (row + 1) / grid_size
            overlap_y = overlap_length(y0, y1, cell_y0, cell_y1)
            if overlap_y <= 0.0:
                continue
            for col in range(col_start, col_end + 1):
                cell_x0 = col / grid_size
                cell_x1 = (col + 1) / grid_size
                overlap_x = overlap_length(x0, x1, cell_x0, cell_x1)
                if overlap_x <= 0.0:
                    continue
                overlap_area = overlap_x * overlap_y
                power_map[row, col] += power_density * overlap_area

    return power_map


def normalize_power_map(power_map: np.ndarray, reference_total_power: float) -> np.ndarray:
    total_power = float(power_map.sum())
    if total_power <= 0.0:
        return power_map
    scale = reference_total_power / total_power
    return power_map * scale


def main():
    args = parse_args()
    deepoheat_root = Path(args.deepoheat_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    layout_path = Path(args.input).resolve()

    add_deepoheat_to_path(deepoheat_root)

    from src import dataio_deeponet, file_parser, modules

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout input not found: {layout_path}")

    with layout_path.open("r", encoding="utf-8") as infile:
        layout = json.load(infile)

    chiplets = layout.get("chiplets", [])
    if not chiplets:
        raise ValueError("Thermal layout does not contain any chiplets")

    power_map = rasterize_power_map(chiplets, grid_size=20)
    power_map = normalize_power_map(power_map, args.reference_total_power)

    sensor = file_parser.convert_interval_to_grid(power_map)
    sensor = sensor / 0.00625

    dataset = build_dataset(dataio_deeponet)
    eval_data = dataset.eval()

    device = "cpu"
    model = build_model(modules, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    coords = eval_data["coords"]
    eval_data = {key: value.float().to(device) for key, value in eval_data.items()}
    eval_data["beta"] = torch.tensor(
        sensor.reshape(1, -1).repeat(coords.shape[0], 0),
        device=device,
    ).float()

    with torch.no_grad():
        temperature = model(eval_data)["model_out"].detach().cpu().numpy().reshape(-1)

    temperature = 293.15 + 25.0 * temperature
    peak_temperature = float(np.max(temperature))
    peak_temperature_c = peak_temperature - 273.15
    print(f"THERMAL_RESULT {peak_temperature:.6f}")
    print(f"THERMAL_RESULT_C {peak_temperature_c:.6f}")


if __name__ == "__main__":
    main()
