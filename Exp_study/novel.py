from random import shuffle, choice, sample, uniform, randint, choices, random
from copy import deepcopy
import time
import csv

def is_valid(individual):
    items_count=[sum(item) for item in list(zip(*individual))]
    if items_count!=list(item_quantities.values()): #is all the demand satisfied?
        return False
    for gene in individual:
        total_length = sum(gene[i] * item_sizes[i] for i in range(len(gene)))
        if total_length > max(stock_lengths): #does the gene exceed the maximum stock length?
            return False
    return True

def evaluate_cost(individual):
    cost=0
    # stocks=[]
    #Get the minimum stock length
    for pattern in individual:
        total_length = sum(pattern[i]*item_sizes[i] for i in range(len(pattern)))
        min_stock_length = min(stock_lengths, key=lambda x: x if x >= total_length else float('inf'))
        # stocks.append(min_stock_length)
        cost += stock_lengths[min_stock_length]
    # print(stocks)
    return cost
def solution_representation(individual):
    representation=[]
    for pattern in individual:
        pat=[]
        total_length = sum(pattern[i]*item_sizes[i] for i in range(len(pattern)))
        min_stock_length = min(stock_lengths, key=lambda x: x if x >= total_length else float('inf'))
        items=[]
        for i in range(len(pattern)):
            if pattern[i]!=0:
                items.extend([item_sizes[i]]*pattern[i])
        pat.append((min_stock_length, items))
        representation.append(pat)
        print(pat)
        # cost += stock_lengths[min_stock_length]
    return representation
    # return cost


def evaluate_fitness(individual):
    return 1/evaluate_cost(individual)

def first_fit_heuristic():
    individual=[]
    orders=[]
    for size, quantity in item_quantities.items():
        orders.extend([size]*quantity)
    shuffle(orders)
    for size in orders:
        for gene in individual:
            sum_of_existing_items=sum([gene[i]*item_sizes[i] for i in range(len(gene))])
            if (sum_of_existing_items+size) <= min(stock_lengths, key=lambda x: x if x >= (sum_of_existing_items+size) else float('inf')):
                gene[item_sizes.index(size)]+=1
                break
        else:
            new_gene=[0]*len(item_sizes)
            new_gene[item_sizes.index(size)]=1
            individual.append(new_gene)
    return individual

def first_fit_decreasing_heuristic(item_quantities,individual):
    # individual = []
    # Sort items by size in decreasing order
    orders=[]
    for size, quantity in item_quantities.items():
        orders.extend([size]*quantity)
    orders = sorted(orders, reverse=True)
    for size in orders:
        for gene in individual:
            sum_of_existing_items=sum([gene[i]*item_sizes[i] for i in range(len(gene))])
            if (sum_of_existing_items+size) <= min(stock_lengths, key=lambda x: x if x >= (sum_of_existing_items+size) else float('inf')):
                gene[item_sizes.index(size)]+=1
                break
        else:
            new_gene=[0]*len(item_sizes)
            new_gene[item_sizes.index(size)]=1
            individual.append(new_gene)
    return individual

def create_initial_population(population_size):
    population=[]
    for _ in range(population_size):
        population.append(first_fit_heuristic())
    return population

def population__str__(population):
    for individual in population:
        print(individual)
        print("\n")

def tournament_selection(population, n_winners, tounament_size=10):
    '''Select parents by tournament selection'''
    parents = []
    for _ in range(n_winners):
        tournament=sample(population, k=tounament_size)
        cheapest_solution = min(tournament, key=lambda x: evaluate_cost(x))
        parents.append(cheapest_solution)
    return parents
def roulette_wheel_selection(population, n_winners):
    total_fitness = sum([evaluate_fitness(individual) for individual in population])
    probabilities = [evaluate_fitness(individual) / total_fitness for individual in population]
    return choices(population, weights=probabilities, k=n_winners)

def novel_recombination(parents, g):
    # Calculate the mean number of cutting patterns
    mean_patterns = sum(len(individual) for individual in parents) / len(parents)
    g = int(mean_patterns + g * mean_patterns) # g as a percentage of mean patterns
    offspring = []
    residual_demand = item_quantities.copy()
    attempt=0
    max_attempts=len(parents)*2
    cur_offspring_size=0
    # Global Recombination
    #While the number of cutting patterns (gene) in the offspring is less than g and there is still residual demand 
    #and the attempt made to add a valid cutting pattern is less than the maximum attempts
    while len(offspring) < g and any(residual_demand.values()) and attempt<max_attempts:
        parent = choice(parents) #Select a random parent
        gene = choice(parent) #Select a random cutting pattern (gene) from the parent
        # Check if the selected gene produces an item for which the demand hasn't been satisfied 
        #and all the items in the gene don't exceed the residual demand
        if all(residual_demand[item_sizes[i]] >= gene[i] for i in range(len(gene))):
            gene = deepcopy(gene) #Deep copy the gene to avoid modifying the parent
            offspring.append(gene)
            cur_offspring_size=len(offspring)
            attempt=0 #Reset the attempt counter
            # Update the residual demand
            for i in range(len(gene)):
                residual_demand[item_sizes[i]] -= gene[i]
        if cur_offspring_size==len(offspring):
            attempt+=1 #Increment the attempt counter if the offspring size hasn't changed
            if attempt==max_attempts:
                #No valid cutting pattern was found in the last max_attempts attempts
                break
    # Constructive Heuristic FFD for residual demand
    if any(residual_demand.values()):
        offspring = first_fit_decreasing_heuristic(residual_demand,offspring)

    return offspring

# item_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
# item_quantities= {3: 5, 4: 2, 5: 1, 6: 2, 7: 4, 8: 2, 9: 1, 10: 3}
# stock_lengths={10: 100, 13: 130, 15: 150}


# item_sizes=[2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# item_quantities= {2350: 2, 2250: 4, 2200: 4, 2100: 15, 2050: 6, 2000: 11, 1950: 6, 1900: 15, 1850: 13, 1700: 5, 1650: 2, 1350: 9, 1300: 3, 1250: 6, 1200: 10, 1150: 4, 1100: 8, 1050: 3}
# stock_lengths={4300: 86, 4250: 85, 4150: 83, 3950: 79, 3800: 68, 3700: 66, 3550: 64, 3500: 63}

item_sizes=[21, 22, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66, 67]
item_quantities= {21: 13, 22: 15, 24: 7, 25: 5, 27: 9, 29: 9, 30: 3, 31: 15, 32: 18, 33: 17, 34: 4, 35: 17, 38: 20, 39: 9, 42: 4, 44: 19, 45: 4, 46: 12, 47: 15, 48: 3, 49: 20, 50: 14, 51: 15, 52: 6, 53: 4, 54: 7, 55: 5, 56: 19, 57: 19, 59: 6, 60: 3, 61: 7, 63: 20, 65: 5, 66: 10, 67: 17}
stock_lengths={120: 12, 115: 11.5, 110: 11, 105: 10.5, 100: 10}


data=[['Generation evolved','Time limit', 'Best Cost', 'Generation Found', 'Time Found']]
generation_costs=[]
# for iteration in range(30):
time_found = 0
time_limit = 60
start_time = time.time()
population_size = 10
population = create_initial_population(population_size)
N_OFFSPRING = len(population)
costs=[evaluate_cost(individual) for individual in population]
best_cost = min(costs)
print(best_cost)
best_individual = population[costs.index(best_cost)]

generation = 0
generation_found = 0
# n_generations = 60

# generation_costs.append([f"{time.time() - start_time}",f"{generation}"]+costs)
while time.time() - start_time < time_limit:
# while generation < n_generations:
# while True:
    generation += 1
    parents=tournament_selection(population, N_OFFSPRING,5)
    # parents=roulette_wheel_selection(population, N_OFFSPRING)

    offspring=novel_recombination(parents, 0.1)
    offspring_cost=evaluate_cost(offspring)
    population.sort(key=lambda x: evaluate_cost(x))
    if offspring_cost<=evaluate_cost(population[-1]):
        population[-1] = offspring
        if offspring_cost < best_cost:
            best_cost = offspring_cost
            best_individual = offspring
            generation_found = generation
            time_found = time.time()-start_time
            print(best_cost)
    generation_costs.append([f"{time.time() - start_time}",f"{generation}"]+[evaluate_cost(individual) for individual in population])
# print(f"Best cost: {best_cost} found in generation {generation_found}")# with individual {best_individual}")
# solution_representation(best_individual)
# print(item_sizes)
# data.append([generation, time_limit, best_cost, generation_found, f"{time_found:.1f}"])
# print(iteration)
# print(generation_found)

# print(generation_costs)

# # # print(solution_representation(individual))
# with open(r'Exp_study\novel_pop100.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(generation_costs)
# print(generation)
