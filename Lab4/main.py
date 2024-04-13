import random
import math
import matplotlib.pyplot as plt
import time

class EvolutionaryAlgorithm:
    def __init__(self, population_size, num_generations, crossover_prob, mutation_prob,
                 renewal_rate, generations_until_renewal):
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.renewal_rate = renewal_rate
        self.generations_until_renewal = generations_until_renewal
        self.generations_until_renewal_actual = generations_until_renewal
        self.best_fitness = float('inf')  # Initialize best fitness to infinity
        self.best_generation = 0  # Initialize the generation where the best fitness is found
        self.best_individual = None  # Initialize the best individual

    def initialize_population(self, size):
        return [[random.uniform(-1, 1) for _ in range(size)] for _ in range(self.population_size)]

    def fitness_function(self, solution):
        total = sum(abs(solution[i]) ** (i + 2) for i in range(len(solution)))
        return total

    def select_parents(self, population, size):
        parents = random.sample(population, size)
        parents.sort(key=lambda x: self.fitness_function(x))
        return parents[0]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, individual, mutation_prob):
        for i in range(len(individual)):
            if random.random() < mutation_prob:
                individual[i] = random.uniform(-1, 1)
        return individual

    def select_survivors(self, population, size):
        population.sort(key=lambda x: self.fitness_function(x))
        return population[:size]

    def renew_population(self, population, size):
        for _ in range(size):
            population[random.randint(0, len(population) - 1)] = [random.uniform(-1, 1) for _ in range(len(population[0]))]
        return population

    def adjust_probabilities(self, generation):
        if generation % self.generations_until_renewal == 0 and generation > 0:
            self.mutation_prob *= 2

    def evolve(self, problem_size):
        for _ in range(10):
            start_time = time.time()  # Start measuring execution time
            population = self.initialize_population(problem_size)
            best_fitness_per_generation = []
            best_individual_per_generation = []
            for generation in range(self.num_generations):
                new_population = []
                for _ in range(self.population_size):
                    parent1 = self.select_parents(population, 2)
                    parent2 = self.select_parents(population, 2)
                    if random.random() < self.crossover_prob:
                        child1, child2 = self.crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1[:], parent2[:]
                    child1 = self.mutation(child1, self.mutation_prob)
                    child2 = self.mutation(child2, self.mutation_prob)
                    new_population.extend([child1, child2])
                population = self.select_survivors(new_population, self.population_size)
                self.adjust_probabilities(generation)
                if generation % self.renewal_rate == 0 and generation > 0:
                    population = self.renew_population(population, int(self.population_size * self.renewal_rate))
                best_fitness = min([self.fitness_function(individual) for individual in population])
                if best_fitness < self.best_fitness:  # Update best fitness and generation if found
                    self.best_fitness = best_fitness
                    self.best_generation = generation + 1
                    self.best_individual = [individual for individual in population if self.fitness_function(individual) == best_fitness][0]
                best_fitness_per_generation.append(best_fitness)
                best_individual_per_generation.append(self.best_individual)
                print(f"Generation {generation + 1}: Best fitness: {best_fitness}, Best individual: {self.best_individual}")
            end_time = time.time()  # Stop measuring execution time
            print(f"Total execution time: {end_time - start_time} seconds")
            return best_fitness_per_generation, best_individual_per_generation

def plot_evolution_fitness(best_fitness_per_generation):
    plt.plot(best_fitness_per_generation)
    plt.title('Evolution of the best fitness over generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()


def main():
    problem_size = int(input("Enter the problem size: "))
    population_size = 100
    num_generations = 10000
    crossover_prob = 0.9 # Increased for more exploitation
    mutation_prob = 0.09999 # Reduced for less exploration
    renewal_rate = 0.00001
    generations_until_renewal = 15000
    algorithm = EvolutionaryAlgorithm(population_size, num_generations, crossover_prob, mutation_prob, renewal_rate,
                                      generations_until_renewal)
    best_fitness_per_generation, best_individual_per_generation = algorithm.evolve(problem_size)

    # Calculate the average fitness
    average_fitness = sum(best_fitness_per_generation) / len(best_fitness_per_generation)
    plot_evolution_fitness(best_fitness_per_generation)
    print(f"Absolute best fitness: {algorithm.best_fitness:.12f} found in generation {algorithm.best_generation}")
    print(f"Average fitness: {average_fitness:.7f}")


if __name__ == "__main__":
    main()
