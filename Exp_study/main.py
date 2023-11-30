from random import shuffle, choice, sample, uniform, randint, choices, random
from copy import deepcopy
import time
##CSP Problem




stock_lengths={10: 100, 13: 130, 15: 150}
requested_pieces= {3: 5, 4: 2, 5: 1, 6: 2, 7: 4, 8: 2, 9: 1, 10: 3}
requested_pieces_quantities= [5, 2, 1, 2, 4, 2, 1, 3]


# requested_pieces=[2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# requested_pieces_quantities= [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]
# stock_lengths={4300: 86, 4250: 85, 4150: 83, 3950: 79, 3800: 68, 3700: 66, 3550: 64, 3500: 63}
# requested_pieces= {2350: 2, 2250: 4, 2200: 4, 2100: 15, 2050: 6, 2000: 11, 1950: 6, 1900: 15, 1850: 13, 1700: 5, 1650: 2, 1350: 9, 1300: 3, 1250: 6, 1200: 10, 1150: 4, 1100: 8, 1050: 3}

pieces=[]
for i, n in requested_pieces.items():
    pieces.extend([i]*n)
# print(pieces)
# [3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10, 10]


def first_fit(stock_lengths, pieces):
    '''This forms the pieces into groups known as genes
    Each gene is a list of pieces that can fit into a stock length
    The stock length isn't recorded; it is assumed that the minimum stock length that fits the gene is used'''
    # Sort pieces in decreasing order
    shuffle(pieces)
    # pieces.sort(reverse=True)
    groups = []

    for item in pieces:
        # Try to fit the item into an existing group
        placed = False
        for group in groups:
            # If you can fit this item into a group and the new group is still within any stock length, add it
            if sum(group) + item <= min(stock_lengths, key=lambda x: x if x >= sum(group) + item else float('inf')):
                group.append(item)
                placed = True
                break
        
        # If the item doesn't fit in any existing groups, create a new group
        if not placed:
            groups.append([item])
    return groups

def get_min_stock_length(group):
    '''Get the minimum stock length that can fit a group'''
    min_stock=min(stock_lengths, key=lambda x: x if x >= sum(group) else float('inf'))
    if min_stock<sum(group): return None
    return min_stock

def evaluate_cost(chromosome):
    '''Calculate the cost of a chromosome(group of genes)'''
    total_cost=0
    for gene in chromosome:
        min_stock=min(stock_lengths, key=lambda x: x if x >= sum(gene) else float('inf')) #Find the minimum stock length that can fit the gene
        total_cost+=stock_lengths[min_stock] #Add the cost of the stock length to the total cost
    return total_cost

def satisfies_all_orders(chromosome):
    '''Ensure the chromosome satisfies all the pieces'''
    pieces_in_chromosome = []
    for x in chromosome:
        pieces_in_chromosome.extend(x)
    if sorted(pieces_in_chromosome) != sorted(pieces):   print("INVALID SOLUTION FOUND")
    return sorted(pieces_in_chromosome) == sorted(pieces)
#Select parents

def select_parents_by_tournament(population, n_winners, tounament_size=5):
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

def rank_based_selection(population, n_winners):
    '''
    The rank based selection selects n_winners by ranking the population by fitness and selecting parents by their rank
    return: list of parents selected by rank based selection
    '''
    n=len(population)
    ranked_population=sorted(population, key=lambda x:invert_cost_to_fitness( evaluate_cost(x)), reverse=True) #Sort the population by fitness with the best fitness first
    ranks=list(range(1,n+1)) #List of ranks
    c=2
    probabilities=[(c - (c*(i / n))) / n for i in ranks] #Calculate the probability of each rank
    return choices(ranked_population, weights=probabilities, k=n_winners) #Select n_winners from parents using the ranked population with set probability of each rank

def mutate(chromosome, mutation_rate):
    '''Mutate a chromosome by rearranging the pieces in two genes'''
    mutated_chromosome=deepcopy(chromosome)
    if uniform(0,1)<mutation_rate:
        #Select two genes to swap
        gene1, gene2=sample(chromosome, k=2)
        pieces=gene1+gene2
        shuffle(pieces)
        groups=[]
        for piece in pieces:
            placed=False
            for group in groups:
                if sum(group)+piece<=min(stock_lengths, key=lambda x: x if x >= sum(group)+piece else float('inf')):
                    group.append(piece)
                    placed=True
                    break
            if not placed:
                groups.append([piece])
        mutated_chromosome.remove(gene1)
        mutated_chromosome.remove(gene2)
        mutated_chromosome.extend(groups)
    return mutated_chromosome

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
    crossover_point =randint(1, min(len(parent1), len(parent2)) - 1)
    # print(crossover_point)
    # print(crossover_point)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

    return offspring1, offspring2
def two_point_crossover(parent1, parent2):
    crossover_point1 =randint(1, len(parent1)//2 - 1)
    crossover_point2 =randint(crossover_point1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
    offspring2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
    return offspring1, offspring2
def uniform_crossover(parent1, parent2):
    offspring1, offspring2 = [], []
    min_length=min(len(parent1), len(parent2))
    for i in range(min_length):
        if random()< 0.5:
            offspring1.append(parent1[i])
            offspring2.append(parent2[i])
        else:
            offspring1.append(parent2[i])
            offspring2.append(parent1[i])
    if len(parent1)>min_length:
        offspring1.extend(parent1[min_length:])
    if len(parent2)>min_length:
        offspring2.extend(parent2[min_length:])
    return offspring1, offspring2
    
def gga_crossover(parent1, parent2):
    '''This selects two random crossing sites and injects the contents of the crossing section of 1st parent at the first crossing site into the 2nd parent and vice versa
    Then it removes the duplicates in the offspring'''
    #Select two random crossing sites for each parent
    parent1=deepcopy(parent1)
    parent2=deepcopy(parent2)
    crossover_point1 =randint(1, len(parent1) - 1)
    crossover_point2 =randint(1, len(parent2) - 1)

    #Inject the contents of the crossing section of 1st parent at the first crossing site into the 2nd parent and vice versa
    offspring1 = parent2[:crossover_point2] + parent1[crossover_point1:] + parent2[crossover_point2:]
    offspring2 = parent1[:crossover_point1] + parent2[crossover_point2:] + parent1[crossover_point1:]

    #Remove the items that are in excess of the required quantities
    for offspring in [offspring1, offspring2]:
        current_counts = {piece: 0 for piece in requested_pieces}
        for gene in offspring:
            new_group = []
            for item in gene:
                if current_counts[item] < requested_pieces[item]:
                    new_group.append(item)
                    current_counts[item] += 1
            if new_group!=[]:
                gene[:] = new_group  # Modify group in place
    # if [] in offspring1: 
    #     print(f"P1: {parent1}") 
    #     print(f"P2: {parent2}")
    #     print(crossover_point1, crossover_point2)
    return offspring1, offspring2

def survival_selection(population, offspring):
    '''This function selects the survivors from the population and the offspring'''
    #Select the best solutions from the population and the offspring
    new_population=[]
    new_population.extend(offspring)
    # #Select the best solutions from the population
    # population.sort(key=lambda x: evaluate_cost(x),)
    # new_population.extend(population[:len(population)//2])
    # #Select the best solutions from the offspring
    # offspring.sort(key=lambda x: evaluate_cost(x))
    # new_population.extend(offspring[:len(offspring)//2])
    return new_population

    
def count_pieces_in_chromosome(chromosome):
    '''This function counts the quantities of pieces in a chromosome'''
    chromosome_pieces_count={}
    for gene in chromosome:
        for piece in gene:
            chromosome_pieces_count[piece]=chromosome_pieces_count.get(piece,0)+1
    return chromosome_pieces_count
def get_shortages_excesses(chromosome):
    '''This function gets the surpluses of pieces in a chromosome'''
    chromosome_pieces_count=count_pieces_in_chromosome(chromosome)
    shortages=[]
    excesses=[]
    for req_piece, req_count in requested_pieces.items():
        count=chromosome_pieces_count.get(req_piece,0)
        if count<req_count:
            shortages.extend([req_piece]*(req_count-count))
        elif count>req_count:
            excesses.extend([req_piece]*(count-req_count))
    return shortages, excesses
            
    return surpluses
def repair_chromosome(chromosome):
    '''This function repairs a chromosome by ensuring the chromosome satisfies all the pieces'''
    '''If there is an excess of a piece, find the group with the highest stock length for better optimization'''
    '''If there is a shortage of a piece, find the least cost group where the piece can fit and add it to the group otherwise add the piece as a new group'''
    '''Note: adding of shortage can be improved by allowing expansion of the group to fit any longer stock length'''
    '''The excess is deliberately handled before the shortage because the excess can be removed to allow the shortage, optimizing the use of the stock length'''
    shortages, excesses=get_shortages_excesses(chromosome)   
    if shortages==[] and excesses==[]: return chromosome
    adjusted_chromosome=deepcopy(chromosome)
    for piece in excesses:
        #If there is an excess of a piece, find the group with the highest stock length for better optimization
        max_stock=0
        max_stock_gene=None
        for gene in adjusted_chromosome:
            if piece in gene:
                stock=get_min_stock_length(gene) #Find the minimum stock length that can fit the gene
                if stock>max_stock:
                    max_stock=stock
                    max_stock_gene=gene
        max_stock_gene.remove(piece) #Remove the piece from the group
        if max_stock_gene==[]: adjusted_chromosome.remove(max_stock_gene)
    for piece in shortages:
        #If there is a shortage of a piece, find a group where the piece can fit with the least cost
        min_cost=float('inf')
        min_cost_gene=None
        for gene in adjusted_chromosome:
            # min_stock=get_min_stock_length(gene) #Find the original minimum stock length that fits the gene
            # if sum(gene)+piece<=min_stock: #If this piece and the gene can fit that original stock length
            min_stock=get_min_stock_length(gene+[piece]) #Find the minimum stock length that can fit the existing gene and the piece
            if min_stock!=None: #If the gene and new piece can fit original stock length or a longer stock length
                cost=stock_lengths[min_stock] #Find the cost of the stock length
                if cost<min_cost:
                    min_cost=cost
                    min_cost_gene=gene
        if min_cost_gene!=None: #If there is a group where the piece can fit
            min_cost_gene.append(piece) #Add the piece to the group
        else: #If there is no group where the piece can fit
            adjusted_chromosome.append([piece]) #Add the piece as a new group
            # print("New group")
    return adjusted_chromosome


def init_population(population_size):
    for _ in range(population_size):
        # solution = generate_solution(stock_lengths, pieces)
        # print(evaluate_cost(solution))
        # population.append(solution)
        chromosome = first_fit(stock_lengths, pieces)
        population.append(chromosome)
population=[]
population_size=10
N_OFFSPRING=population_size
init_population(population_size)

least_cost= min([evaluate_cost(x) for x in population])
least_cost_chromosome= [x for x in population if evaluate_cost(x)==least_cost][0]

time_limit=5
start=time.time()
cur_min=least_cost
while time.time()-start<time_limit:
        # print(chromosome,cost)
        # if cost<=4000:
        #     print(chromosome, cost)
        # print(list(map(lambda x: get_min_stock_length(x), chromosome))) 
    parents=select_parents_by_tournament(population, N_OFFSPRING)
    # parents=select_parents_by_roulette(population, N_OFFSPRING)
    parents=rank_based_selection(population, N_OFFSPRING)
    for p in parents:
        print(p)
    print("\n")
    children=[]
    for i in range(0,N_OFFSPRING,2):
        offspring1,offspring2=one_point_crossover(parents[i], parents[i+1])
        # offspring1,offspring2=two_point_crossover(parents[i], parents[i+1])
        # offspring1,offspring2=uniform_crossover(parents[i], parents[i+1])
        offspring1,offspring2=gga_crossover(parents[i], parents[i+1])
        # validated_offspring1=repair_chromosome([[4, 3, 7], [10], [9, 6], [10], [8], [4], [7], [3], [7], [8], [4], [7]])
        # validated_offspring1=repair_chromosome([[10, 3], [3, 3], [8, 7], [6, 5], [7, 8], [9], [6], [10], [7, 4, 3], [10], [3, 3], [8, 7], [6, 5], [7, 8], [9], [6]])
        validated_offspring1=repair_chromosome(offspring1)
        validated_offspring2=repair_chromosome(offspring2)
        # validated_offspring1, validated_offspring2=offspring1, offspring2
        validated_offspring1=mutate(validated_offspring1, 0.7)
        validated_offspring2=mutate(validated_offspring2, 0.7)

        # satisfies_all_orders(validated_offspring1)
        # satisfies_all_orders(validated_offspring2)
        # print(f"Parent 1: {parents[i]}")
        # print(f"Parent 2: {parents[i+1]}")
        # print(f"Offspring 1: {validated_offspring1}")
        # print(f"Offspring 2: {validated_offspring2}\n")
        for ofs in [validated_offspring1,validated_offspring2]:
            cost=evaluate_cost(ofs)
            if cost<least_cost:
                least_cost=cost
                least_cost_chromosome=ofs
        generation_min=min([evaluate_cost(x) for x in population])
        if generation_min!=cur_min:
            cur_min=generation_min
            print(cur_min)
        children.append(validated_offspring1)
        children.append(validated_offspring2)
    population.clear()
    population.extend(survival_selection(parents, children))
# print(min([evaluate_cost(x) for x in population]))
'''My novel idea could be based on the following:
1. The selection of the parents is making the algorithm converge to certain solutions, 
 I need to find a selector operator to make the algorithm select parents that are not similar to each other
'''
# for ch in children:
#     print(ch, evaluate_cost(ch))
print(least_cost_chromosome, least_cost)

    # print(population[i],population[i+1])
    # print(offspring1,offspring2)
    # print(o1,o2)
    # print(evaluate_cost(population[i]),evaluate_cost(population[i+1]))
    # print(evaluate_cost(o1),evaluate_cost(o1))
    # print("\n")
# print(population[0])
# print(population[1])
# print(test)
# print(satisfies_all_orders(test[0]), satisfies_all_orders(test[1]))

# print(f"Initialized {len(population)} population min: {min([evaluate_cost(x) for x in population])}, max: {max([evaluate_cost(x) for x in population])}")
# selected_parents = select_parents_by_tournament(population, n_winners=len(population))
# print(f"Tournament {len(selected_parents)} min: {min([evaluate_cost(x) for x in selected_parents])}, max: {max([evaluate_cost(x) for x in selected_parents])}")
# selected_parents = select_parents_by_roulette(population, n_winners=len(population))
# print(f"Roulette {len(selected_parents)} min: {min([evaluate_cost(x) for x in selected_parents])}, max: {max([evaluate_cost(x) for x in selected_parents])}")

# print(selected_parents[0])
# print("\n")
# print(selected_parents[1])

# # distinct_solutions={}
# # for x in selected_parents:
# #     distinct_solutions[tuple(sorted(x.items()))]=distinct_solutions.get(tuple(sorted(x.items())),0)+1
# # print(len(distinct_solutions))
# print("\n")
# print(one_point_crossover(selected_parents[0], selected_parents[1]))


# # print(all([satisfies_all_orders(x) for x in population]))

# # # print(list(map(lambda x :x.values(), population)))
# # costs=list(map(lambda x :evaluate_cost(x), population))
# # print(costs)
# # print(min(costs))





'''
After generating offsprings by crossover (by finding the midpoints in the two parents and swapping those)
Repair function to fix the new solution'''

'''Optimum solution: 
[[7, 3, 3], [10, 3], [9, 6], [7, 4, 3], [8, 4, 3], [10, 5], [7, 6], [8, 7], [10]] 1240
[[6, 4, 5], [7, 8], [10, 3], [9, 6], [10, 3], [7, 7], [8, 7], [3, 10], [3, 4, 3]] 1240
[[3, 8, 4], [8, 3, 4], [9, 6], [10, 3], [10, 5], [6, 7], [7, 7], [3, 10], [3, 7]] 1240
[[9, 3, 3], [6, 3, 3, 3], [8, 7], [7, 8], [10, 5], [4, 7, 4], [7, 6], [10], [10]] 1230
'''



'''
3988
[[1200, 1250, 1250], [1250, 1100, 1100], [1100, 1250, 1200], [2000, 1300], [1150, 2350], [2000, 2250], [1950, 1850], [1950, 1850], [1950, 1850], [2050, 1650], [1950, 1850], [1850, 1950], [2200, 1300], [1350, 2000], [1650, 2000], [2100, 1200], [2250, 1150], [1200, 2200], [2250, 1250], [2100, 2050], [2200, 1200], [2100, 2050], [2100, 1700], [1900, 1900], [1100, 1050, 1200], [1900, 1900], [2000, 1700], [1900, 1900], [1900, 1900], [2350, 1200], [2250, 1200], [1900, 1900], [1350, 2000], [2100, 1350], [2050, 2100], [2100, 1350], [1100, 2000, 1050], [2100, 1350], [2050, 2100], [1900, 1900], [1700, 2000], [1850, 1850], [1850, 1850], [1850, 1850], [2100, 1200], [1050, 1350, 1100], [1850, 1900], [1100, 1100, 2000], [2100, 1350], [2000, 1700], [1900, 1900], [1350, 2100], [1950, 1850], [1350, 2100], [1300, 2200], [2100, 2050], [2000, 1700], [2100, 1250], [1150, 1200, 1150]]'''