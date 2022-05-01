"""Generates instance inputs of small, medium, and large sizes.

Modify this file to generate your own problem instances.

For usage, run `python3 generate.py --help`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict

from instance import Instance
from size import Size
from point import Point
from file_wrappers import StdoutFileWrapper

import random

def gen_cities(prob_type):
    assert prob_type in ["small", "medium", "large"], "prob_type must be small, medium, or large"

    D_dict = {"small": 30, "medium": 50, "large": 100}
    N_bounds_dict = {"small": (15, 25), "medium": (45, 55), "large": (195, 205)}
    rp_dict = {"small": 8, "medium": 10, "large": 14}

    N_upper, N_lower = N_bounds_dict[prob_type]
    N = random.randint(N_upper, N_lower)
    D = D_dict[prob_type]
    rp = rp_dict[prob_type]

    placed = set()
    while len(placed) < N:
        pos = (random.randint(0, D-1), random.randint(0, D-1))
        placed.add(pos)
    
    assert len(placed) == N

    cities = []
    for pos in placed:
        x, y = pos
        cities.append(Point(x, y))
    
    return cities 


def make_small_instance() -> Instance:
    """Creates a small problem instance.

    Size.SMALL.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    cities = []
    # YOUR CODE HERE
    cities = gen_cities("small")
    return Size.SMALL.instance(cities)


def make_medium_instance() -> Instance:
    """Creates a medium problem instance.

    Size.MEDIUM.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    cities = []
    # YOUR CODE HERE
    cities = gen_cities("medium")
    return Size.MEDIUM.instance(cities)


def make_large_instance() -> Instance:
    """Creates a large problem instance.

    Size.LARGE.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    cities = []
    # YOUR CODE HERE
    cities = gen_cities("large")
    return Size.LARGE.instance(cities)


# You shouldn't need to modify anything below this line.
SMALL = 'small'
MEDIUM = 'medium'
LARGE = 'large'

SIZE_STR_TO_GENERATE: Dict[str, Callable[[], Instance]] = {
    SMALL: make_small_instance,
    MEDIUM: make_medium_instance,
    LARGE: make_large_instance,
}

SIZE_STR_TO_SIZE: Dict[str, Size] = {
    SMALL: Size.SMALL,
    MEDIUM: Size.MEDIUM,
    LARGE: Size.LARGE,
}


def outfile(args, size: str):
    if args.output_dir == "-":
        return StdoutFileWrapper()

    return (Path(args.output_dir) / f"{size}.in").open("w")


def main(args):
    for size, generate in SIZE_STR_TO_GENERATE.items():
        if size not in args.size:
            continue

        with outfile(args, size) as f:
            instance = generate()
            assert instance.valid(), f"{size.upper()} instance was not valid."
            assert SIZE_STR_TO_SIZE[size].instance_has_size(instance), \
                f"{size.upper()} instance did not meet size requirements."
            print(f"# {size.upper()} instance.", file=f)
            instance.serialize(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate problem instances.")
    parser.add_argument("output_dir", type=str, help="The output directory to "
                        "write generated files to. Use - for stdout.")
    parser.add_argument("--size", action='append', type=str,
                        help="The input sizes to generate. Defaults to "
                        "[small, medium, large].",
                        default=None,
                        choices=[SMALL, MEDIUM, LARGE])
    # action='append' with a default value appends new flags to the default,
    # instead of creating a new list. https://bugs.python.org/issue16399
    args = parser.parse_args()
    if args.size is None:
        args.size = [SMALL, MEDIUM, LARGE]
    main(args)
