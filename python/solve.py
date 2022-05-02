"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
from turtle import position
from typing import Callable, Dict, List
from turtle import pen
import random

from instance import Instance
from solution import Solution
from Individual import Individual
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
from point import Point


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def solve_GA(instance: Instance) -> Solution:
    population_size = 100000
    population = []

    for _ in range(population_size):
        ind = Individual.create_individual(instance)
        population.append(ind)

    num_generations = 100

    while (num_generations > 0):
        population = sorted(population, key=lambda x: x.fitness)
        new_generation = []

        for individual in population:
            if individual.fitness == -1000:
                population.remove(individual)

        s = int((10*len(population))/100)
        new_generation.extend(population[:s])

        s = int((90*len(population))/100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        if (len(population)) < 10:
            num_generations = 0
            break

        num_generations -= 1

    population = sorted(population, key = lambda x : x.fitness)

    best_individual = population[0]

    print(f'Penalty: {best_individual.fitness}')

    return best_individual.solution



def num_cities_covered(pos: Point, cities: dict, coverage_radius: int) -> int:
    covered = 0
    for city in cities:
        if city.distance_sq(pos) <= coverage_radius:
            covered += 1
    return covered

def num_towers_conflicting(new_tower: Point, placed_towers: List[Point], penalty_radius: int) -> int:
    conflicts = 0
    for placed_tower in placed_towers:
        if new_tower.distance_sq(placed_tower) <= penalty_radius:
            conflicts += 1
    return conflicts

def utility_func(pos: Point, cities: dict, towers: list, coverage_radius: int, penalty_radius: int, ncw = 2.5, ntw = 0.5):
    # return num_cities_covered(pos, cities, coverage_radius)
    return ncw*num_cities_covered(pos, cities, coverage_radius) - ntw*num_towers_conflicting(pos, towers, penalty_radius)

def get_tower_bounding_box(pos: Point) -> List[Point]:
    x, y = pos.x, pos.y

    bb_ = [(x-3, y),(x-2, y),(x-1, y),(x+1, y),(x+2, y),(x+3, y),
    (x, y-3),(x, y-2),(x, y-1),(x, y+1),(x, y+2),(x, y+3),
    (x-1, y+2), (x, y+2), (x+1, y+2),
    (x-2, y+1), (x-1, y+1), (x, y+1), (x+1, y+1), (x+2, y+1),
    (x-1, y-2), (x, y-2), (x+1, y-2),
    (x-2, y-1), (x-1, y-1), (x, y-1), (x+1, y-1), (x+2, y-1)]
    bb = [Point(p[0], p[1]) for p in bb_]

    return bb

def solve_greedy(instance: Instance) -> Solution:
    cities = instance.cities
    grid_side_length = instance.grid_side_length
    coverage_radius = instance.coverage_radius
    penalty_radius = instance.penalty_radius
    N = len(cities)
    coverage_radius_sq = coverage_radius ** 2
    penalty_radius_sq = penalty_radius ** 2

    num_trials = 5

    cities = [city for city in cities]
    placed_towers = []
    positions = set([Point(x, y) for x in range(grid_side_length) for y in range(grid_side_length)
                    if num_cities_covered(Point(x,y), cities, coverage_radius_sq) != 0])

    utility = lambda pos: utility_func(pos, cities, placed_towers, coverage_radius_sq, penalty_radius_sq, ncw = 2.5, ntw = 0.5)
    utility_early_stopping = lambda pos: utility_func(pos, cities, placed_towers, coverage_radius_sq, penalty_radius_sq, ncw = 1, ntw = 0)

    flag = False

    while len(cities) > 0:

        search_nums = [int((1 - (0.45*len(cities)/N)) * len(positions)) for _ in range(num_trials)]

        search_positions = [random.sample(positions, search_nums[i]) for i in range(num_trials)]

        if len(placed_towers) >= 0.90 * N or flag:
            max_towers = [max(search_positions[i], key = lambda pos: utility_early_stopping(pos)) for i in range(num_trials)]
            tower_to_place = max(max_towers, key = lambda pos: utility_early_stopping(pos))
        else:
            max_towers = [max(search_positions[i], key = lambda pos: utility(pos)) for i in range(num_trials)]
            tower_to_place = max(max_towers, key = lambda pos: utility(pos))

        original_cities_length = len(cities)
        temp_cities = list(filter(lambda city: city.distance_sq(tower_to_place) > coverage_radius_sq, cities))
        new_cities_length = len(temp_cities)
        if original_cities_length == new_cities_length:
            flag = True
            continue

        cities = temp_cities
        placed_towers.append(tower_to_place)
        positions.remove(tower_to_place)
        positions.difference_update(set(get_tower_bounding_box(tower_to_place)))

        # print(f"NUM CITIES UNCOVERED: {len(cities)}")
        # print(f"NUM TOWER PLACED: {len(placed_towers)}")

    sol = Solution(instance=instance, towers=placed_towers)
    print(f"PENALTY: {sol.penalty()}")
    return sol


def coverage(pos: Point, cities: List, coverage_radius: int) -> int:
    num_covered = 0
    for city in cities:
        if city.distance_sq(pos) <= (coverage_radius ** 2):
            num_covered += 1
    return num_covered

def update_coverage(pos: Point, cities: List, coverage_radius: int) -> int:
    new_cities = []
    for city in cities:
        if city.distance_sq(pos) <= (coverage_radius ** 2):
            continue
        new_cities.append(city)
    return new_cities

def penalty_heuristic(new_tower: Point, placed_towers: List[Point], penalty_radius: int) -> int:
    num_conflicts = 0
    for placed_tower in placed_towers:
        if new_tower.distance_sq(placed_tower) <= (penalty_radius ** 2):
            num_conflicts += 1
    return num_conflicts

def greedy_utility(pos: Point, uncovered_cities: List[Point], placed_towers: List[Point], coverage_radius: int, penalty_radius: int) -> int:
    ncw = 2.5
    ntw = 0.5

    city_coverage = coverage(pos, uncovered_cities, coverage_radius)
    tower_penalties = penalty_heuristic(pos, placed_towers, penalty_radius)

    return (ncw * city_coverage) - (ntw * tower_penalties)

def in_bounds(pos: Point, grid_boundry: int) -> bool:
    if (pos.x < 0 or pos.y < 0):
        return False
    if (pos.x >= grid_boundry or pos.y >= grid_boundry):
        return False
    return True

def prune_search(placed_tower, other_towers, positions, penalty_radius, grid_boundary):
    penalty_bound = 2 * penalty_radius

    search_space = set()
    for bound_x in range(placed_tower.x - penalty_bound, placed_tower.x + penalty_bound + 1):
        upper_SP = Point(bound_x, placed_tower.y + penalty_bound)
        upper_conflicts = penalty_heuristic(upper_SP, other_towers, penalty_radius)
        lower_SP = Point(bound_x, placed_tower.y - penalty_bound)
        lower_conflicts = penalty_heuristic(upper_SP, other_towers, penalty_radius)

        if (in_bounds(upper_SP, grid_boundary) and (upper_SP in positions) and upper_conflicts == 0):
            search_space.add(upper_SP)
            # positions.remove(upper_SP)

        if (in_bounds(lower_SP, grid_boundary) and (lower_SP in positions) and lower_conflicts == 0):
            search_space.add(lower_SP)
            # positions.remove(lower_SP)

    for bound_y in range(placed_tower.y - penalty_bound, placed_tower.y + penalty_bound + 1):
        upper_SP = Point(placed_tower.x + penalty_bound, bound_y)
        upper_conflicts = penalty_heuristic(upper_SP, other_towers, penalty_radius)
        lower_SP = Point(placed_tower.x - penalty_bound, bound_y)
        lower_conflicts = penalty_heuristic(upper_SP, other_towers, penalty_radius)


        if (in_bounds(upper_SP, grid_boundary) and (upper_SP in positions) and upper_conflicts == 0):
            search_space.add(upper_SP)
            # positions.remove(upper_SP)

        if (in_bounds(lower_SP, grid_boundary) and (lower_SP in positions) and lower_conflicts == 0):
            search_space.add(lower_SP)
            # positions.remove(lower_SP)

    return search_space

def search(placed_towers, positions, penalty_radius, grid_boundary):
    final_search = set()
    for tower in placed_towers:
        our_search = prune_search(tower, placed_towers, positions, penalty_radius, grid_boundary)
        final_search = final_search.union(our_search)
    return final_search

def solve_greedy_opt(instance: Instance) -> Solution:
    placed_towers = []
    uncovered_cities = instance.cities
    positions = set([Point(x, y) for x in range(instance.grid_side_length) for y in range (instance.grid_side_length)
        if coverage(Point(x, y), uncovered_cities, instance.coverage_radius) != 0])
    utility = lambda pos: greedy_utility(pos, uncovered_cities, placed_towers, instance.coverage_radius, instance.penalty_radius)

    tower_to_place = max(positions, key = lambda pos : utility(pos))
    placed_towers.append(tower_to_place)
    search_space = search(placed_towers, positions, instance.penalty_radius, instance.grid_side_length)
    uncovered_cities = update_coverage(tower_to_place, uncovered_cities, instance.coverage_radius)

    while len(uncovered_cities) > 0:
        if len(search_space) == 0:
            tower_to_place = max(positions, key = lambda pos : utility(pos))
        else:
            tower_to_place = max(search_space, key = lambda pos : utility(pos))

        positions.remove(tower_to_place)
        placed_towers.append(tower_to_place)
        search_space = search(placed_towers, positions, instance.penalty_radius, instance.grid_side_length)
        uncovered_cities = update_coverage(tower_to_place, uncovered_cities, instance.coverage_radius)

        print(f'Positions Length : {len(positions)}')
        print(f'Search Space : {len(search_space)}')

    sol = Solution(instance=instance, towers=placed_towers)
    print(f"PENALTY: {sol.penalty()}")

    return sol

    
SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy,
    "genetic" : solve_GA,
    "greedy_opt" : solve_greedy_opt
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