import random
import math
from dataclasses import dataclass
from typing import List, Tuple


Point = Tuple[float, float]


@dataclass
class LinearParams:
    a: float
    b: float


@dataclass
class QuadraticParams:
    a: float
    b: float
    c: float


@dataclass
class ExpParams:
    a: float
    b: float


def sample_linear(x: float, p: LinearParams) -> float:
    return p.a * x + p.b


def sample_quadratic(x: float, p: QuadraticParams) -> float:
    return p.a * x * x + p.b * x + p.c


def sample_exp(x: float, p: ExpParams) -> float:
    return p.a * math.exp(p.b * x)


def noisy(y: float, sigma: float) -> float:
    return y + random.gauss(0.0, sigma)


def random_outlier(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Point:
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    return (x, y)


def generate_points_for_model(
        model_name: str,
        count: int,
        x_range: Tuple[float, float],
        noise_sigma: float,
        linear_params: LinearParams,
        quadratic_params: QuadraticParams,
        exp_params: ExpParams,
) -> List[Point]:
    points: List[Point] = []

    for _ in range(count):
        x = random.uniform(*x_range)

        if model_name == "linear":
            y = sample_linear(x, linear_params)
        elif model_name == "quadratic":
            y = sample_quadratic(x, quadratic_params)
        elif model_name == "exp":
            y = sample_exp(x, exp_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y = noisy(y, noise_sigma)
        points.append((x, y))

    return points


def split_counts(total: int, ratios: Tuple[int, int, int], outlier_ratio: float) -> Tuple[int, int, int, int]:
    outliers = round(total * outlier_ratio)
    clean_total = total - outliers

    s = sum(ratios)
    c1 = round(clean_total * ratios[0] / s)
    c2 = round(clean_total * ratios[1] / s)
    c3 = clean_total - c1 - c2

    return c1, c2, c3, outliers


def shuffle_points(points: List[Point]) -> List[Point]:
    pts = points[:]
    random.shuffle(pts)
    return pts


def generate_mixed_dataset(
        total_points: int,
        dominant_model: str,
        dominant_ratio: int,
        minor_ratio_1: int,
        minor_ratio_2: int,
        outlier_ratio: float = 0.1,
        x_range: Tuple[float, float] = (-3.0, 3.0),
        noise_sigma: float = 0.8,
        linear_params: LinearParams = LinearParams(2.0, 1.0),
        quadratic_params: QuadraticParams = QuadraticParams(1.2, -0.5, 2.0),
        exp_params: ExpParams = ExpParams(1.5, 0.6),
        y_outlier_range: Tuple[float, float] = (-40.0, 40.0),
) -> List[Point]:
    if dominant_model not in {"linear", "quadratic", "exp"}:
        raise ValueError("dominant_model must be one of: linear, quadratic, exp")

    others = [m for m in ["linear", "quadratic", "exp"] if m != dominant_model]

    ratios_map = {
        dominant_model: dominant_ratio,
        others[0]: minor_ratio_1,
        others[1]: minor_ratio_2,
    }

    c_dom, c_o1, c_o2, c_out = split_counts(
        total_points,
        (dominant_ratio, minor_ratio_1, minor_ratio_2),
        outlier_ratio,
    )

    counts_map = {
        dominant_model: c_dom,
        others[0]: c_o1,
        others[1]: c_o2,
    }

    points: List[Point] = []

    for model_name in ["linear", "quadratic", "exp"]:
        cnt = counts_map.get(model_name, 0)
        points.extend(
            generate_points_for_model(
                model_name=model_name,
                count=cnt,
                x_range=x_range,
                noise_sigma=noise_sigma,
                linear_params=linear_params,
                quadratic_params=quadratic_params,
                exp_params=exp_params,
            )
        )

    for _ in range(c_out):
        points.append(random_outlier(x_range, y_outlier_range))

    return shuffle_points(points)


def save_points_txt(points: List[Point], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for x, y in points:
            f.write(f"{x:.6f}, {y:.6f}\n")


def save_points_csv(points: List[Point], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in points:
            f.write(f"{x:.6f},{y:.6f}\n")


def print_points(points: List[Point]) -> None:
    for x, y in points:
        print(f"{x:.6f}, {y:.6f}")


def generate_three_test_samples(
        total_points: int = 100,
        dominant_ratio: int = 70,
        minor_ratio_1: int = 15,
        minor_ratio_2: int = 15,
        outlier_ratio: float = 0.1,
        x_range: Tuple[float, float] = (-3.0, 3.0),
        noise_sigma: float = 0.8,
) -> Tuple[List[Point], List[Point], List[Point]]:
    sample_linear_dominant = generate_mixed_dataset(
        total_points=total_points,
        dominant_model="linear",
        dominant_ratio=dominant_ratio,
        minor_ratio_1=minor_ratio_1,
        minor_ratio_2=minor_ratio_2,
        outlier_ratio=outlier_ratio,
        x_range=x_range,
        noise_sigma=noise_sigma,
        linear_params=LinearParams(2.2, -1.0),
        quadratic_params=QuadraticParams(0.9, 0.7, 1.5),
        exp_params=ExpParams(1.2, 0.55),
        y_outlier_range=(-50.0, 50.0),
    )

    sample_quadratic_dominant = generate_mixed_dataset(
        total_points=total_points,
        dominant_model="quadratic",
        dominant_ratio=dominant_ratio,
        minor_ratio_1=minor_ratio_1,
        minor_ratio_2=minor_ratio_2,
        outlier_ratio=outlier_ratio,
        x_range=x_range,
        noise_sigma=noise_sigma,
        linear_params=LinearParams(-1.7, 3.0),
        quadratic_params=QuadraticParams(1.4, -0.8, 2.5),
        exp_params=ExpParams(1.0, 0.45),
        y_outlier_range=(-60.0, 60.0),
    )

    sample_exp_dominant = generate_mixed_dataset(
        total_points=total_points,
        dominant_model="exp",
        dominant_ratio=dominant_ratio,
        minor_ratio_1=minor_ratio_1,
        minor_ratio_2=minor_ratio_2,
        outlier_ratio=outlier_ratio,
        x_range=x_range,
        noise_sigma=noise_sigma,
        linear_params=LinearParams(1.5, 0.5),
        quadratic_params=QuadraticParams(0.8, 0.2, 1.0),
        exp_params=ExpParams(1.8, 0.7),
        y_outlier_range=(-40.0, 80.0),
    )

    return sample_linear_dominant, sample_quadratic_dominant, sample_exp_dominant


if __name__ == "__main__":
    random.seed(123456)

    sample1, sample2, sample3 = generate_three_test_samples(
        total_points=120,
        dominant_ratio=70,
        minor_ratio_1=15,
        minor_ratio_2=15,
        outlier_ratio=0.12,
        x_range=(-2.5, 2.5),
        noise_sigma=0.9,
    )

    save_points_txt(sample1, "../../data/hw003/sample_linear_dominant.txt")
    save_points_txt(sample2, "../../data/hw003/sample_quadratic_dominant.txt")
    save_points_txt(sample3, "../../data/hw003/sample_exp_dominant.txt")

    save_points_csv(sample1, "../../data/hw003/sample_linear_dominant.csv")
    save_points_csv(sample2, "../../data/hw003/sample_quadratic_dominant.csv")
    save_points_csv(sample3, "../../data/hw003/sample_exp_dominant.csv")

    print("Generated:")
    print("  sample_linear_dominant.txt / .csv")
    print("  sample_quadratic_dominant.txt / .csv")
    print("  sample_exp_dominant.txt / .csv")