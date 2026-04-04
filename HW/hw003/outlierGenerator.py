import random


def generate_outliers(
        n=80,
        xmin=-15.0,
        xmax=20.0,
        ymin=-30.0,
        ymax=40.0,
        seed=None
):
    if seed is not None:
        random.seed(seed)

    for _ in range(n):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        print(f"{x:.1f}, {y:.1f}")


if __name__ == "__main__":
    generate_outliers()
