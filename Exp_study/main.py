from random import shuffle, choice, sample, uniform, randint
from copy import deepcopy
##CSP Problem



stock_lengths=[10, 13, 15]
stock_costs=[100, 130, 150]
pieces_lengths=[3, 4, 5, 6, 7, 8, 9, 10]
pieces_quantities= [5, 2, 1, 2, 4, 2, 1, 3]

# stock_lengths=[4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
# stock_costs=[86, 85, 83, 79, 68, 66, 64, 63]
# pieces_lengths=[2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# pieces_quantities= [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]
pieces=[]
for i, n in zip(pieces_lengths, pieces_quantities):
    pieces.extend([i]*n)
# print(pieces)
# [3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10, 10]

def generate_activity(stock_lengths, remaining_pieces):
    stock_length = choice(stock_lengths) #pick a random stock length
    activity = []
    total_length = 0

    while remaining_pieces and total_length < stock_length:
        piece = choice(remaining_pieces) #pick a random piece
        if total_length + piece > stock_length: #if the piece is too big to fit in the stock
            possible_pieces = [p for p in remaining_pieces if total_length + p <= stock_length] #Find all the pieces that can fit in the stock
            if possible_pieces:
                piece = choice(possible_pieces)
            else: #if no piece can fit in the stock, stop
                break
        activity.append(piece)
        total_length += piece
        remaining_pieces.remove(piece)

    return stock_length, activity

def generate_solution(stock_lengths, pieces):
    solution = {}
    remaining_pieces = pieces.copy()

    while remaining_pieces:
        stock_length, activity = generate_activity(stock_lengths, remaining_pieces)
        if activity:  # Only add non-empty activities
            # solution.append((stock_length, activity))
            activity = tuple(sorted(activity))
            solution[(stock_length, activity)] = solution.get((stock_length, activity), 0) + 1
    return solution


def evaluate_cost(solution):
    '''Calculate the cost of a solution'''
    total_cost = sum([stock_costs[stock_lengths.index(activity[0])]*cost for activity, cost in solution.items()])
    return total_cost
def satisfies_all_orders(solution):
    '''Ensure the solution satisfies all the pieces'''
    pieces_in_solution = []
    for x in solution:
        activity= list(x[1])
        count = solution[x]
        pieces_in_solution.extend(activity*count)
    return sorted(pieces_in_solution) == sorted(pieces)

#Select parents
def select_parents_by_tournament(population, n_winners, tounament_size=10):
    '''Select parents by tournament selection'''
    parents = []
    for _ in range(n_winners):
        tournament=sample(population, k=tounament_size)
        cheapest_solution = min(tournament, key=lambda x: evaluate_cost(x))
        parents.append(cheapest_solution)
    return parents

def select_parents_by_roulette(population, n_winners):
    '''Select parents by roulette wheel selection'''
    parents = []
    fitness_scores=[invert_cost_to_fitness(evaluate_cost(solution)) for solution in population]
    fittest_solution = max(fitness_scores)
    for _ in range(n_winners):
        spin= uniform(0,fittest_solution)
        for i, fitness in enumerate(fitness_scores):
            if fitness_scores[i]>=spin:
                parents.append(population[i])
                break
    return parents
def invert_cost_to_fitness(cost):
        return 1/cost  

# def crossover(parents):
#     '''Crossover the parents to generate offsprings'''
#     offspring = {}
#     for parent in parents:
#         for activity, count in parent.items():
#             offspring[activity] = offspring.get(activity, 0) + count
#     return offspring

def one_point_crossover(parent1, parent2):
    '''
    This function performs one point crossover between two parents
    i.e it takes a random point in the parents and swaps the tails of the parents'''
    # Convert dictionaries to lists
    list1 = list(parent1.items())
    list2 = list(parent2.items())

    # Choose crossover point
    crossover_point =randint(1, min(len(list1), len(list2)) - 1)

    # Create offsprings
    offspring1 = dict(list1[:crossover_point] + list2[crossover_point:])
    offspring2 = dict(list2[:crossover_point] + list1[crossover_point:])
    return offspring1, offspring2


population=[]
for _ in range(50):
    solution = generate_solution(stock_lengths, pieces)
    population.append(solution)


print(f"Initialized {len(population)} population min: {min([evaluate_cost(x) for x in population])}, max: {max([evaluate_cost(x) for x in population])}")
selected_parents = select_parents_by_tournament(population, n_winners=len(population))
print(f"Tournament {len(selected_parents)} min: {min([evaluate_cost(x) for x in selected_parents])}, max: {max([evaluate_cost(x) for x in selected_parents])}")
# selected_parents = select_parents_by_roulette(population, n_winners=len(population))
# print(f"Roulette {len(selected_parents)} min: {min([evaluate_cost(x) for x in selected_parents])}, max: {max([evaluate_cost(x) for x in selected_parents])}")

print(selected_parents[0])
print("\n")
print(selected_parents[1])

# distinct_solutions={}
# for x in selected_parents:
#     distinct_solutions[tuple(sorted(x.items()))]=distinct_solutions.get(tuple(sorted(x.items())),0)+1
# print(len(distinct_solutions))
print("\n")
print(one_point_crossover(selected_parents[0], selected_parents[1]))


# print(all([satisfies_all_orders(x) for x in population]))

# # print(list(map(lambda x :x.values(), population)))
# costs=list(map(lambda x :evaluate_cost(x), population))
# print(costs)
# print(min(costs))

# # Activitiy-based representation



'''
After generating offsprings by crossover (by finding the midpoints in the two parents and swapping those)
Repair function to fix the new solution'''
# def generate_cutting_patterns(stock_length, orders):
#     def backtrack(current_combination, remaining_length, start):
#         if current_combination:   result.add(Activity(stock_length,tuple(sorted(current_combination)))) #() (10)
#         for i in range(start,len(orders)): #0 - 11 0 - 11
#             if orders[i] <= remaining_length: #10 <= 40
#                 backtrack(current_combination + [orders[i]], remaining_length - orders[i],i+1)
#     result = set()
#     backtrack([], stock_length,0)
#     return sorted(result, key=lambda x: sum(x.recipes))
# def evaluate_cost(solution):
#     '''Calculate the cost of a solution'''
#     total_cost = sum([activity.stock_cost for activity in solution])
#     return total_cost
# def is_valid(activity):
#     if sum(activity.recipes) > activity.stock_length:
#         return False
#     return True
# # print(f"Stock lengths: {stock_lengths}, Pieces: {pieces}")
# for l in stock_lengths:
#     # print(f"Stock length: {l}")
#     activities.extend(generate_cutting_patterns(l, pieces))
#     # print(all([is_valid(x, l) for x in length_activities]))
#     # print(set(tuple(x) for x in length_activities))
#     # print(f"Activities: {length_activities}")
#     # print(list(map(lambda x: x.recipes,length_activities)))
#     # break
# # print(activities)

# def generate_solutions(n=100):
#     '''A solution is a dictionary of an activity as key and its quantity as value (i.e how many times it is repeated)'''
#     '''All the recipes of all the activities in a solution put togeter is the required pieces'''
#     population=[]
#     for _ in range(n):
#         solution={}
#         remaining_pieces = deepcopy(pieces)
#         while remaining_pieces: #until all pieces are satisfied
#             random_activity = choice(activities)
#             #if all the recipes of the activity are in the remaining pieces and the count each recipe is less than the quantity of the recipe
#             if all([remaining_pieces.count(x) and remaining_pieces.count(x) >= random_activity.recipes.count(x) for x in random_activity.recipes]):
#                 solution[random_activity] = solution.get(random_activity, 0) + 1 #add the activity to the solution or increment its quantity if it is already in the solution
#                 for r in random_activity.recipes:  #remove the recipes of the activity from the remaining pieces
#                     remaining_pieces.remove(r)
#         population.append(solution) 
#     return population

# population=generate_solutions()
# print(population)
# # for solution in population:
# #     test=[]
# #     for act, q in solution.items():
# #         test.extend(list(act.recipes)*q)
# #     print(sorted(test))
# # # cutting_pattern=
# # print([f"{x.recipes}-{q}" for x, q in solution.items()])
# # print(solution)#
# # print(evaluate_cost(solution))
# # print(len(activities))
# # print(all([is_valid(x) for x in activities]))
