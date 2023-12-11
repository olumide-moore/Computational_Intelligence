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

def one_point_crossover(parent1, parent2):
    # Select a random crossover point
    crossover_point = randint(1, len(parent1) - 1)
    # Create children by crossover
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def uniform_crossover(parent1, parent2):
    child1, child2 = [], []
    min_length=min(len(parent1), len(parent2))
    for i in range(min_length):
        if random()< 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    if len(parent1)>min_length:
        child1.extend(parent1[min_length:])
    if len(parent2)>min_length:
        child2.extend(parent2[min_length:])
    return child1, child2

def mutate(individual, mutation_rate=0.7):
    mutated_individual = deepcopy(individual)
    if uniform(0, 1) <= mutation_rate:
        gene1, gene2 = sample(mutated_individual, 2) # Select two random genes
        i = randint(0, len(gene1) - 1) # Select a random position in the genes
        gene1[i], gene2[i] = gene2[i], gene1[i] # Swap quantities in the genes
    return mutated_individual

def repair_offspring(offsp):
    '''Repair given offspring to make it feasible'''
    '''Ensure sum of items in each gene doesn't exceed maximum stock length'''
    '''Remove surplus quantities and add missing quantities'''
    offspring = deepcopy(offsp)
    # Check if the sum of items in the offspring genes doesn't exceed stock length
    for gene in offspring:
        total_length = sum(gene[i] * item_sizes[i] for i in range(len(gene)))
        while total_length > max(stock_lengths):
            #Remove items from the gene until it is valid
            for i in range(len(gene)):
                if gene[i]!=0:
                    gene[i]-=1
                    break
            total_length = sum(gene[i] * item_sizes[i] for i in range(len(gene)))
    # Calculate current quantities of each item
    current_quantities = {size: sum(gene[item_sizes.index(size)] for gene in offspring) for size in item_sizes}

    # Adjust surplus and deficit
    while True:
        surplus = {size: current_quantities[size] - item_quantities[size]
                   for size in item_sizes if current_quantities[size] > item_quantities[size]}
        deficit = {size: item_quantities[size] - current_quantities[size]
                   for size in item_sizes if current_quantities[size] < item_quantities[size]}
        if not surplus and not deficit:
            break  # The solution is feasible
        # Adjust surplus by reducing excess items in the genes
        for size, excess in surplus.items():
            for gene in offspring:
                if gene[item_sizes.index(size)] > 0:
                    remove_qty = min(gene[item_sizes.index(size)], excess)
                    gene[item_sizes.index(size)] -= remove_qty
                    excess -= remove_qty
                    if excess == 0:
                        break

        # Adjust deficit by adding missing items to the genes
        for size, needed in deficit.items():
            for gene in offspring:
                if can_add_item_to_gene(gene, size):
                    add_qty = min(needed, available_space_in_gene(gene, size))
                    gene[item_sizes.index(size)] += add_qty
                    needed -= add_qty
                    if needed == 0:
                        break
            else:
                # No gene can accommodate the item, so add a new gene
                for _ in range(needed):
                    gene = [0] * len(item_sizes)
                    gene[item_sizes.index(size)] = 1
                    offspring.append(gene)
        # Update current quantities
        current_quantities = {size: sum(gene[item_sizes.index(size)] for gene in offspring) for size in item_sizes}
    # if [0]*len(item_sizes) in offspring:
    #     offspring.remove([0]*len(item_sizes))
    return offspring

def can_add_item_to_gene(gene, size):
    total_length = sum(gene[i] * item_sizes[i] for i in range(len(gene)))
    return total_length + size <= min(stock_lengths, key=lambda x: x if x >= total_length + size else float('inf'))

def available_space_in_gene(gene, item_size):
    total_length = sum(gene[i] * item_sizes[i] for i in range(len(gene)))
    available_space = max(stock_lengths) - total_length
    return available_space // item_size

# item_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
# item_quantities= {3: 5, 4: 2, 5: 1, 6: 2, 7: 4, 8: 2, 9: 1, 10: 3}
# stock_lengths={10: 100, 13: 130, 15: 150}


# item_sizes=[2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# item_quantities= {2350: 2, 2250: 4, 2200: 4, 2100: 15, 2050: 6, 2000: 11, 1950: 6, 1900: 15, 1850: 13, 1700: 5, 1650: 2, 1350: 9, 1300: 3, 1250: 6, 1200: 10, 1150: 4, 1100: 8, 1050: 3}
# stock_lengths={4300: 86, 4250: 85, 4150: 83, 3950: 79, 3800: 68, 3700: 66, 3550: 64, 3500: 63}


item_sizes=[21, 22, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66, 67]
item_quantities= {21: 13, 22: 15, 24: 7, 25: 5, 27: 9, 29: 9, 30: 3, 31: 15, 32: 18, 33: 17, 34: 4, 35: 17, 38: 20, 39: 9, 42: 4, 44: 19, 45: 4, 46: 12, 47: 15, 48: 3, 49: 20, 50: 14, 51: 15, 52: 6, 53: 4, 54: 7, 55: 5, 56: 19, 57: 19, 59: 6, 60: 3, 61: 7, 63: 20, 65: 5, 66: 10, 67: 17}
stock_lengths={120: 12, 115: 11.5, 110: 11, 105: 10.5, 100: 10}
# # pieces=[]
# # for i, n in requested_pieces.items():
# #     pieces.extend([i]*n)
# # # print(pieces)


data=[['Generations','Time taken', 'Best Cost', 'Generation Found', 'Time Found', 'Repair Time']]
generation_costs=[]

# for iteration in range(30):
time_found = 0
time_limit = 60
start_time = time.time()
population_size = 10
population = create_initial_population(population_size)
N_OFFSPRING = len(population)
costs=[evaluate_cost(individual) for individual in population]
# generation_costs.append(costs)
best_cost = min(costs)
print(best_cost)
# print(costs)
best_individual = population[costs.index(best_cost)]
repair_sum=0
generation = 0
generation_found = 0
generation_costs.append([f"{time.time() - start_time}",f"{generation}"]+costs)
while time.time() - start_time < time_limit:
    generation += 1
    parents=tournament_selection(population, N_OFFSPRING,5)
    # parents=roulette_wheel_selection(population, N_OFFSPRING)
    offspring=[]
    for i in range(0,len(parents),2):
        offspring1,offspring2=one_point_crossover(parents[i],parents[i+1])
        # offspring1,offspring2=uniform_crossover(parents[i],parents[i+1])
        off1, off2 = mutate(offspring1,0.7), mutate(offspring2,0.7)
        # repair_time=time.time()
        off1, off2 = repair_offspring(off1), repair_offspring(off2)
        # repair_sum+=(time.time()-repair_time)
        if not is_valid(off1) or not is_valid(off2):
            print("Invalid individual found!")
        offspring.append(off1)
        offspring.append(off2)
    for i in range(len(offspring)):
        if evaluate_cost(offspring[i]) < best_cost:
            best_cost = evaluate_cost(offspring[i])
            # if best_cost==1230:
            #     print(time.time()-start_time)
            #     break
            best_individual = offspring[i]
            generation_found = generation
            time_found = time.time()-start_time
            print(best_cost)
    population.clear()
    population.extend(offspring)
    # print([evaluate_cost(individual) for individual in population])
    generation_costs.append([f"{time.time() - start_time}",f"{generation}"]+[evaluate_cost(individual) for individual in population])
        
# print(f"Best cost: {best_cost} found in generation {generation_found}")# with individual {best_individual}")
# # print(item_sizes)
# data.append([generation, time_limit, best_cost, generation_found, f"{time_found:.1f}", f"{repair_sum:.1f}"])
# print(iteration)
# print(data)
# print(repair_sum)
with open(r'Exp_study\baseline_costs3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(generation_costs)