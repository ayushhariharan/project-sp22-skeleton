import enum
import random
from solution import Solution
from point import Point

class Individual(object):
    def __init__(self, solution, num_towers, problem) -> None:
        self.solution = solution
        self.num_towers = num_towers
        self.problem = problem
        self.fitness = self.calc_fitness()
    
    def create_individual(problem_instance):
        N = problem_instance.grid_side_length
        num_towers = random.randint(1, len(problem_instance.cities))

        tower_positions = set()
        while len(tower_positions) < num_towers:
            pos = Point(random.randint(0, N-1), random.randint(0, N-1))
            tower_positions.add(pos)

        new_solution = Solution(
            instance=problem_instance,
            towers=list(tower_positions)
        )

        return Individual(solution=new_solution, num_towers=num_towers, problem=problem_instance)

    def mate(self, individual_2):
        largest_tower = []
        smallest_tower = []
        min_towers = -1

        if (self.num_towers > individual_2.num_towers):
            largest_tower = self.solution.towers
            smallest_tower = individual_2.solution.towers
            min_towers = individual_2.num_towers
        else:
            largest_tower = individual_2.solution.towers
            smallest_tower = self.solution.towers
            min_towers = self.num_towers
        
        child_towers = []
        for (i, pos) in enumerate(largest_tower):
            selection_probability = random.random()

            if (selection_probability < 0.45):
                child_towers.append(pos)
            elif (i < min_towers):
                new_pos = smallest_tower[i]
                child_towers.append(new_pos)

        num_towers = len(child_towers)
        child_solution = Solution(
            instance=self.problem,
            towers=child_towers
        )

        return Individual(solution=child_solution, num_towers=num_towers, problem=self.problem)
    
    def calc_fitness(self):
        if self.solution.valid():
            return self.solution.penalty()
        return -1000