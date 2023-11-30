from random import shuffle, choice, sample, uniform, randint, choices, random
from copy import deepcopy
import time

# def generate_individual(order_demands,stock_lengths):
#     individual = []
#     remaining_demands = order_demands.copy()
#     while any(remaining_demands.values()):
#         pattern, chosen_stock_length = generate_cutting_pattern(remaining_demands, stock_lengths)
#         individual.append(pattern + [chosen_stock_length])

#         # Update remaining demands
#         for i, size in enumerate(order_demands):
#             remaining_demands[size] = max(0, remaining_demands[size] - pattern[i])

#     return individual

# def generate_cutting_pattern(remaining_demands, stock_lengths):
#     stock_length = choice(list(stock_lengths.keys()))
#     pattern = [0] * len(order_demands)
#     remaining_length = stock_length

#     for i, size in enumerate(order_demands):
#         if size <= remaining_length and remaining_demands[size] > 0:
#             max_count = min(remaining_length // size, remaining_demands[size])
#             count = randint(0, max_count)
#             pattern[i] = count
#             remaining_length -= count * size
#     return pattern, stock_length

def is_valid(individual):
    items_count=[sum(item) for item in list(zip(*individual))]
    return items_count==list(order_demands.values())

def evaluate_cost(individual):
    cost=0
    #Get the minimum stock length
    for pattern in individual:
        total_length = sum(pattern[i]*item_sizes[i] for i in range(len(pattern)))
        min_stock_length = min(stock_lengths, key=lambda x: x if x >= total_length else float('inf'))
        cost += stock_lengths[min_stock_length]
    return cost



# # order_demands = {2: 10, 3: 8, 5: 6}
# # stock_lengths = {10: 100, 15: 150, 20: 200}

# order_demands= {3: 5, 4: 2, 5: 1, 6: 2, 7: 4, 8: 2, 9: 1, 10: 3}
# stock_lengths={10: 100, 13: 130, 15: 150}

# # requested_pieces=[2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# # requested_pieces_quantities= [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]
# # stock_lengths={4300: 86, 4250: 85, 4150: 83, 3950: 79, 3800: 68, 3700: 66, 3550: 64, 3500: 63}
# # order_demands= {2350: 2, 2250: 4, 2200: 4, 2100: 15, 2050: 6, 2000: 11, 1950: 6, 1900: 15, 1850: 13, 1700: 5, 1650: 2, 1350: 9, 1300: 3, 1250: 6, 1200: 10, 1150: 4, 1100: 8, 1050: 3}

# # pieces=[]
# # for i, n in requested_pieces.items():
# #     pieces.extend([i]*n)
# # # print(pieces)

# individual = generate_individual(order_demands, stock_lengths)
# print(individual)
# print(is_valid(individual))
# print(evaluate_cost(individual))

from random import randint, shuffle, choice

def generate_cutting_pattern_RER1():
    individual = []
    # Sort items by size in decreasing order
    orders=[]
    for size, quantity in order_demands.items():
        orders.extend([size]*quantity)
    orders = sorted(orders, reverse=True)
    for size in orders:
        for gene in individual:
            if sum([gene[i]*item_sizes[i] for i in range(len(gene))]) <= min(stock_lengths, key=lambda x: x if x >= sum([gene[i]*item_sizes[i] for i in range(len(gene))]) + size else float('inf')):
                gene[item_sizes.index(size)]+=1
                break
            # if sum(gene) + size <= min(stock_lengths, key=lambda x: x if x >= sum(gene) + size else float('inf')):
            #     gene.append(size)
            #     break
        else:
            var=[0]*len(item_sizes)
            var[item_sizes.index(size)]=1
            individual.append(var)
    return individual

# def generate_cutting_pattern_RER2():
#     # Randomly rank items and determine count based on remaining space and demand percentage
#     orders=deepcopy(item_sizes)
#     shuffle(orders)
#     pattern = []
#     demand_percentage = randint(10, 100) / 100
#     for size in orders:

#     #     max_count = min(stock_length // size, int(order_demands[size] * demand_percentage))
#     #     count = min(max_count, stock_length // size)
#     #     for _ in range(count):
#     #         pattern.append(size)
#     #         stock_length -= size
#     #         order_demands[size] -= 1
#     # return pattern
def generate_cutting_pattern_RER2():
    individual = []
    remaining_demands = order_demands.copy()

    while any(remaining_demands.values()):
        # Randomly rank items
        items = list(remaining_demands.keys())
        shuffle(items)

        gene = [0] * len(item_sizes)
        chosen_stock = choice(list(stock_lengths.keys()))
        remaining_space = chosen_stock

        # Determine the percentage for inclusion in the pattern
        inclusion_percentage = uniform(0.1, 1.0)

        for item in items:
            if remaining_demands[item] > 0:
                qty_to_include = min(remaining_space // item, int(inclusion_percentage * remaining_demands[item]))
                if qty_to_include == 0 and remaining_space >= item:
                    qty_to_include = 1
                gene[item_sizes.index(item)] = qty_to_include
                remaining_space -= qty_to_include * item
                remaining_demands[item] -= qty_to_include
        if sum(gene) > 0:  # Ensure that the gene adds at least one item
            individual.append(gene)

    return individual

def create_initial_population(population_size):
    population = [generate_cutting_pattern_RER1()]
    for _ in range(1, population_size):
        population.append(generate_cutting_pattern_RER2())
    return population

# Example
# stock_lengths = {10: 100, 15: 150}
# item_sizes = [2, 3, 5]
# order_demands = {2: 10, 3: 8, 5: 6}
population_size = 10

item_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
order_demands= {3: 5, 4: 2, 5: 1, 6: 2, 7: 4, 8: 2, 9: 1, 10: 3}
stock_lengths={10: 100, 13: 130, 15: 150}


initial_population = create_initial_population(population_size)
print(initial_population)
for individual in initial_population:
    print(is_valid(individual))
    print(evaluate_cost(individual))
