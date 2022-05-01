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

import heapq

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

def solve_greedy(instance: Instance) -> Solution:
    cities = instance.cities
    grid_side_length = instance.grid_side_length
    coverage_radius = instance.coverage_radius
    penalty_radius = instance.penalty_radius
    num_cities = len(cities)



    heap = []

    temp_cities_covered = 0

    return Solution(
        instance=instance,
        towers=instance.cities
    )



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
