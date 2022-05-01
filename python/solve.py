"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List

from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
from point import Point

import heapq
import math

def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )


def num_cities_covered(pos: Point, cities: dict, coverage_radius: int) -> int:
    covered = 0
    for city in cities:
        if cities[city] == 0 and math.sqrt(city.distance_sq(pos)) <= coverage_radius:
            covered += 1
    return covered

def num_towers_conflicting(new_tower: Point, placed_towers: List[Point], penalty_radius: int) -> int:
    conflicts = 0
    for placed_tower in placed_towers:
        if math.sqrt(new_tower.distance_sq(placed_tower)) <= penalty_radius:
            conflicts += 1
    return conflicts

def utility_func(pos: Point, cities: dict, towers: list, coverage_radius: int, penalty_radius: int, ncw = 0.5, ntw = 0.5):
    return num_cities_covered(pos, cities, coverage_radius) 
    # return ncw*num_cities_covered(pos, cities, coverage_radius) - ntw*num_towers_conflicting(pos, towers, penalty_radius)


def solve_greedy(instance: Instance) -> Solution:
    cities = instance.cities
    grid_side_length = instance.grid_side_length
    coverage_radius = instance.coverage_radius
    penalty_radius = instance.penalty_radius
    num_cities = len(cities)

    cities = {city: 0 for city in cities}
    placed_towers = []
    positions = set([Point(x, y) for x in range(grid_side_length) for y in range(grid_side_length)])

    print(f"NUM CITIES: {num_cities}")
    while sum(cities.values()) < num_cities:
        heap = []
    
        for pos in positions:
            heapq.heappush(heap, (-utility_func(pos, cities, placed_towers, coverage_radius, penalty_radius, ncw=10, ntw=1),
                                    id(pos), pos))

        _, __, tower_to_place = heapq.heappop(heap)

        for city in cities:
            if math.sqrt(city.distance_sq(tower_to_place)) <= coverage_radius:
                cities[city] = 1

        placed_towers.append(tower_to_place)
        positions.remove(tower_to_place)
         
        print(f"NUM CITIES COVERED: {sum(cities.values())}")
        print(f"NUM TOWER PLACED: {len(placed_towers)}")
        print(f"HEAP FIRST ELEM: {heap[0]}")


    return Solution(instance=instance, towers=placed_towers)






SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
