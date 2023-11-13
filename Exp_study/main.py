from random import shuffle
from copy import deepcopy
##CSP Problem

# #Representation of the problem
class Order:
    def __init__(self, length, quantity):
        self.length = length
        self.quantity = quantity
    def __str__(self):
        return "Order(length={}, quantity={})".format(self.length, self.quantity)

class Stock:
    def __init__(self, length, cost):
        self.length = length
        self.cost = cost
    def __str__(self):
        return "Stock(length={}, cost={})".format(self.length, self.cost)

class Activity():
    def __init__(self, stock, recipes):
        self.stock = stock
        self.recipes = recipes

    def is_valid(self):
        '''Ensure the total length of recipes does not exceed the stock length'''
        return sum(self.recipes) <= self.stock.length
    def __str__(self):
        return f"stock={self.stock.__str__()}, recipes={self.recipes}"
    def __repr__(self):
        return f"stock={self.stock.__str__()}, recipes={self.recipes}"

def evaluate_cost(solution):
    '''Calculate the cost of a solution'''
    total_cost = sum([activity.stock.cost for activity in solution])
    return total_cost


def is_valid(recipes, stock):
    '''Ensure the total length of recipes does not exceed the stock length'''
    return sum(recipes) <= stock

def generate_activities(orders, stocks):
    '''Generate possible activities(solution)'''
    '''This picks each stock and tries to fit as many orders (of left orders) into it 
    and repeats this until all orders are assigned to a stock'''
    activities = []

    orders_remaining = deepcopy(orders)

    #Sort stocks by length in descending order
    while orders_remaining:
        for stock in stocks:
            recipes = []
            length_remaining = stock.length
            for i in range(len(orders_remaining)):
                order = orders_remaining[i]
                while order.quantity > 0 and order.length <= length_remaining:
                    recipes.append(order.length)
                    length_remaining -= order.length
                    order.quantity -= 1
            if recipes:#if recipes is not empty, add it to activities
                activities.append(Activity(stock, recipes))
                #Remove orders with quantity 0
                orders_remaining = [x for x in orders_remaining if x.quantity > 0]
    return activities



#Generate activities by randomizing the order and stocks
def random_population(orders, stocks, n):
    '''Generate initial population by randomizing the order and stocks'''
    population = []
    for _ in range(n):
        orders_copy = deepcopy(orders)
        stocks_copy = deepcopy(stocks)
        shuffle(orders_copy)
        shuffle(stocks_copy)
        population.append(generate_activities(orders_copy, stocks_copy))
    return population


stocks = [Stock(10, 100), Stock(13, 130), Stock(15, 150)]
orders= [Order(3, 5), Order(4, 2), Order(5, 1), Order(6, 2), Order(7, 4), Order(8, 2), Order(9, 1), Order(10, 3)]

population=random_population(orders, stocks, 50)
population_cost = [evaluate_cost(solution) for solution in population]
print(min(population_cost), max(population_cost), sum(population_cost)/len(population_cost))
# print(f"\nTotal cost of {list(map(lambda x:x.__str__(),orders_copy)), list(map(lambda x: x.__str__(),stocks_copy))}: {evaluate_cost(activities)} from randomizing the order and stocks")



##Order-based representation
def generate_random_order_based_population(orders,n):
    '''Generate a list of orders from a list of activities'''
    contiguous_orders = []
    for order in orders:
        for _ in range(order.quantity):
            contiguous_orders.append(order.length)
    
    population = []
    for _ in range(n):
        shuffle(contiguous_orders)
        population.append(contiguous_orders)
    return population


#Take each order and try to fit it into stocks that has been cut
#if it can't fit, cut a new stock and add that stock to the list of cut stocks
#This decoding funciton tends to favor the initial stocks in the list when fitting orders
#This can lead to a bias towards the initial stocks over the rest, potentially leading to a local optimum
#especially if the initial stocks are not the best stocks to cut
def decoding_function(orders,stocks):
    '''Decoding function that takes a list of orders and returns a list of activities'''
    '''This picks each order and tries to fit it into stocks that has been cut'''
    activities=[]
    cut_stocks=[] #list of stocks that has been cut. rep: tuple of (stock object, length remaining, recipes)

    for order in orders:
        for stock in cut_stocks:
            if stock[1]>=order:
                stock[1]-=order
                stock[2].append(order)
                break
        else: #if the order cannot fit into any of the cut stocks, cut a new stock
            for stock in stocks:
                if stock.length>=order:
                    cut_stocks.append([stock,stock.length-order,[order]])
                    break
    for stock in cut_stocks:
        activities.append(Activity(stock[0],stock[2]))

    return activities
def decoding_function_by_least_wastage(orders,stocks):
    activities=[]
    cut_stocks=[] #list of stocks that has been cut. rep: tuple of (stock object, length remaining, recipes)

    for order in orders:
        best_stock=None
        for stock in cut_stocks:
            if stock[1]>=order and (best_stock==None or stock[1]-order<best_stock[1]-order):
                best_stock=stock
        if best_stock:
            best_stock[1]-=order
            best_stock[2].append(order)
        else: #if the order cannot fit into any of the cut stocks, cut a new stock
            for stock in stocks:
                if stock.length>=order:
                    cut_stocks.append([stock,stock.length-order,[order]])
                    break
    for stock in cut_stocks:
        activities.append(Activity(stock[0],stock[2]))
    return activities
population=generate_random_order_based_population(orders, 1)
print(population)
for individual in population:
    print(decoding_function_by_least_wastage(individual,stocks))
