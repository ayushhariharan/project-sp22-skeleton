"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict
import random

from instance import Instance
from solution import Solution
from Individual import Individual
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
from point import Point

import heapq
import math

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
    positions = [Point(x, y) for x in range(grid_side_length) for y in range(grid_side_length)]

    while sum(cities.values()) < num_cities:
        
        tower_to_place = positions[0]

        for pos in positions:
            if utility_func(tower_to_place) < utility_func(pos):
                tower_to_place = pos 
            
        for city in cities:
            if math.sqrt(city.distance_sq(tower_to_place)) <= coverage_radius:
                cities[city] = 1

        placed_towers.append(tower_to_place)
        positions.remove(tower_to_place)
         
        print(f"NUM CITIES COVERED: {sum(cities.values())}")
        print(f"NUM TOWER PLACED: {len(placed_towers)}")


    return Solution(instance=instance, towers=placed_towers)






SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy,
    "genetic" : solve_GA
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
