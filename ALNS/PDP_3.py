import csv
import random
import networkx as nx
import os

import pandas as pd
from alns import ALNS,State
from alns.criteria import SimulatedAnnealing
from alns.criteria import HillClimbing
import numpy.random as rnd

import copy
import itertools
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy as np

# construct a graph
os.chdir(r"C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\network")
node = pd.read_csv("node_convert_to_degree.csv")
G = nx.DiGraph()
for i in range(len(node)):
    G.add_node(node.loc[i,'id'], size=0.01,weight = 0,coordinate =(node.loc[i,'Lat'],node.loc[i,'Lon']))
node_num = G.number_of_nodes()
link = pd.read_csv('Riverside_time_updat.csv') # use the update time information with XML+ link_12pm
for j in range(len(link)):
    G.add_edge(link.loc[j,'X_from'],link.loc[j,'X_to'],id = link.loc[j,'X_id'],length = link.loc[j,'X_length'],travel_time = link.loc[j,'X_travel_time'] )


# data preparation
max_order_number = 20 ##每次最多有10单 （p+d）
routing_result = pd.read_csv(r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\routing_result.csv',index_col=False)
task_whole_info = pd.read_csv(r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\task.csv') #already sort by ETA, and assign unique id to each one
task_map = pd.read_csv(r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\task_map.csv',index_col=False) # map of task and point of interest
driver = pd.read_csv(r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\driver_10_unique_id.csv') # driver information also assign unique id to each driver



def distance_time_of_2task(task_1,task_2): ## return travel_distance and travel time between any two points
    routing_pair = routing_result[(routing_result['from']== task_map.loc[task_1,'point of interest']) & (routing_result['to']==task_map.loc[task_2,'point of interest'])] # locate in the routing result
    distance = routing_pair['travel_distance'].values[0] # get series value
    time = routing_pair['travel_time'].values[0]/60 # convert to min
    return distance,time

## determine distance of one route
def distance_time_estimation(unique_id,task_sequence): ## input: a route; output: the total travel distance and distace between two orders, not consider the driver to the first restaurant
    distance_vector = []
    delay_vector = []
    if len(task_sequence)==0: # empty route
        one_route_distance = 0
        one_route_duration = 0
        one_route_delay = 0
    else:
        start_time = task_whole_info.iloc[task_sequence[0]]['EPT'] - 10
        time = task_whole_info.iloc[task_sequence[0]]['EPT'] - 10  # 假设driver 在接第一个订单的前10分钟上班
        for i in range(len(task_sequence) - 1):
            travel_distance,travel_time = distance_time_of_2task(task_sequence[i],task_sequence[i+1])
            distance_vector.append(travel_distance) # record consecutive result
            time = time+travel_time
            if task_sequence[i+1]%2 == 0: # pick_up point
                if time < task_whole_info.iloc[task_sequence[i + 1]]['EPT']:
                    time = task_whole_info.iloc[task_sequence[i+1]]['EPT']
                time += 1  ##service time 1min
            else: # delivery point
                if time > task_whole_info.iloc[task_sequence[i+1]]['ETA']:
                    delay_vector.append(time - task_whole_info.iloc[task_sequence[i+1]]['ETA'])
                else:
                    delay_vector.append(0)
                time += 1 ##service time 1min
        one_route_distance = sum(distance_vector)
        one_route_duration = time - start_time
        one_route_delay = sum(delay_vector)
    return one_route_distance, distance_vector,one_route_duration,one_route_delay,delay_vector

## feasibility control #控制前后，控制由同一辆车送一个订单，控制订单数，控制司机的schedule，控制
def route_cost(unique_id,task_sequence):
    distance, distance_vector, duration, delay, delay_vector = distance_time_estimation(unique_id,task_sequence)
    return alpa*distance + beta*delay


def feasibility_control(unique_id,task_sequence): ## 给入一个 route，判断是否可行
    if len(task_sequence)>=10 & len(task_sequence)<=max_order_number: # 大于10的时候，再cehck duration
        duration = distance_time_estimation(unique_id, task_sequence)[2]
        if duration > 120:
            return False
        else:
            return True
    elif len(task_sequence)> max_order_number:
        return False
    else:
        return True

alpa = 0.3 # weight for distance
beta = 0.7 # weight for delay
# weights for objective function
class PDPState(State): ## given the orders and generated routes, what is the objective value? routes: arrary of route object

    def __init__(self,routes):
        self.routes = routes

    def order_number(self):
        total_order = 0
        for key in self.routes.keys(): # loop over the dictionary
            total_order += len(self.routes[key])/2 # 每个route 的task sequence， then divide by 2
        return total_order

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        distance_total = 0
        delay_total = 0
        #duration = 0
        for key in self.routes:
            distance, distance_vector,duration,delay,delay_vector = distance_time_estimation(key,self.routes[key])
            delay_total += delay
            distance_total += distance
        return alpa*distance_total+beta*delay_total

    def distance_delay(self):
        distance_total = 0
        delay_total = 0
        # duration = 0
        for key in self.routes:
            distance, distance_vector, duration, delay, delay_vector = distance_time_estimation(key, self.routes[key])
            delay_total += delay
            distance_total += distance
        return distance_total,delay_total
    # time waste  route arrival time of the pickup location - EPT
    # time delay. route arrival time of the delivery location - ETA
    # energy consumption: sum of energy consumption of two consecutive points.


# define destroy operator
degree_of_destruction = 0.2
def orders_to_remove(state):
    return int(state.order_number()*degree_of_destruction)

# removal operators##


def random_removal(current,random_state): ##[[],[],[],[]]
    # pool = []
    destroyed = current.copy()
    for i in range(orders_to_remove(destroyed)):
        index_1 = random.choice([*current.routes.keys()])## randomly pick a route_key, these routes are non-empty
        if len(destroyed.routes[index_1])>0:
            task = random.choice(destroyed.routes[index_1])  # randomly chosen a task
            if task%2 ==0: # pick up point
                if task + 1 in destroyed.routes[index_1]:
                    destroyed.routes[index_1].remove(task)
                    destroyed.routes[index_1].remove(task + 1)
                    # pool.append(task)
                    # pool.append(task+1)
            elif task % 2 == 1:  # delivery point
                if task - 1 in destroyed.routes[index_1]:
                    destroyed.routes[index_1].remove(task - 1)
                    destroyed.routes[index_1].remove(task)
                    # pool.append(task-1)
                    # pool.append(task)
            else:
                continue
        else:
            continue
    return destroyed


def longest_segment_removal(current,random_state): # remove the top_N longest segments between two orders
    # pool = []
    destroyed = current.copy()
    for i in range(orders_to_remove(destroyed)):
        average_route_distance = []
        for key,route in destroyed.routes.items():  # find the worst route  distance/n # How to cope with n ==0
            one_route_distance, distance_vector,one_route_duration,one_route_delay,delay_vector = distance_time_estimation(key,route)
            if len(route)>0: # exclude the empty route
                average_route_distance.append(one_route_distance/len(route))
            else:
                average_route_distance.append(0)
        index_1 = average_route_distance.index(max(average_route_distance)) # worst way
        one_route_distance, distance_vector,one_route_duration,one_route_delay,delay_vector = distance_time_estimation(index_1,destroyed.routes[index_1])  # find the worst order in the route
        index_2 = distance_vector.index(max(distance_vector))
        task = destroyed.routes[index_1][index_2] # find the worst task
        if task % 2 == 0:  # pick up point
            if task + 1 in destroyed.routes[index_1]:
                destroyed.routes[index_1].remove(task)
                destroyed.routes[index_1].remove(task + 1)
                # pool.append(task)
                # pool.append(task + 1)
        elif task % 2 == 1:  # delivery point
            if task - 1 in destroyed.routes[index_1]:
                destroyed.routes[index_1].remove(task - 1)
                destroyed.routes[index_1].remove(task)
                # pool.append(task - 1)
                # pool.append(task)
        else:
            continue
    return destroyed

def path_removal(current,random_state): ## remove the longest distance path
    destroyed = current.copy()
    route_distance = []
    for key,route in destroyed.routes.items():
        one_route_distance, distance_vector,one_route_duration,one_route_delay,delay_vector = distance_time_estimation(key,destroyed.routes[key])
        route_distance.append(one_route_distance)
    index_1 = route_distance.index(max(route_distance))
    # pool = destroyed.routes[index_1] ##是否要判断 p+d pair，保证由一个driver完成订单
    destroyed.routes[index_1] = []
    return destroyed

# def longest_delay_removal(current,random_state):
#     pool = []
#     destroyed = current.copy()

# def shaw_removal(current,random_state):
#     destroyed = current.copy()
#
#     return destroyed

## insertion operators ##
def random_repair(current,random_state):
    inserted = list(np.concatenate(list(current.routes.values())).flat)
    pool = list(set(task_pool)-set(inserted))
    pool.sort()
    print('random_pool', pool)
    for i in range(0,len(pool),2):
        index_1 = random.choice([*current.routes.keys()]) # choose a route_key
        k = random.randint(0,len(current.routes[index_1])) # randomly choose a position
        current.routes[index_1].insert(k, pool[i]) # insert provider
        j = random.randint(k+1,len(current.routes[index_1]))  # randomly choose a position
        current.routes[index_1].insert(j, pool[i+1]) # insert customer
    return current

def greedy_insertion(current,random_state):
    inserted = list(np.concatenate(list(current.routes.values())).flat)
    pool = list(set(task_pool)-set(inserted))
    print('greedy_pool',pool)
    while len(pool)>0:
        # print('pool',pool)
        # print('current.routes',current.routes)
        chosen_route,pick_node,insert_index_1,delivery_node,insert_index_2 = findGreedyInsert(current,pool) # find greedy p+d pair and its greedy position
        # print('chosen_route,pick_node,insert_index_1,delivery_node,insert_index_2',chosen_route,pick_node,insert_index_1,delivery_node,insert_index_2)
        current.routes[chosen_route].insert(insert_index_1,pick_node)
        current.routes[chosen_route].insert(insert_index_2,delivery_node)
        pool.remove(pick_node)
        pool.remove(delivery_node)
    return current

def findGreedyInsert(current,pool): # find best orders to be inserted in best position
    best_insert_pickup = 0
    best_insert_delivery = 0
    best_chosen_route = 0
    best_insert_index_1 = 0
    best_insert_index_2 = 0
    best_cost = float('inf')
    for i in range(0,len(pool),2):# all p+d pair try all possible routes and all posible positions in that chosen route
        for key,route in current.routes.items(): # loop over all routes
            for k in range(len(route)+1): # +1, cause it can insert at the end of the route.
                for j in range(k+1,len(route)+2): # +1, cause it can insert at the end of the route.
                    temp_route = copy.deepcopy(route) # only copy that route, not the whole solution
                    temp_route.insert(k, pool[i])
                    temp_route.insert(j, pool[i + 1])
                    if feasibility_control(key, temp_route): # check feasibility fisrt, then calculate objective
                        distance, distance_vector, duration, delay, delay_vector = distance_time_estimation(key,temp_route)
                        delta_cost = alpa*distance+beta*delay
                        if delta_cost < best_cost : # compare with best increase so far, use to pick the best order, and record the minimum postion
                            best_cost = delta_cost
                            best_chosen_route = key
                            best_insert_pickup = pool[i]
                            best_insert_delivery = pool[i+1]
                            best_insert_index_1 = k
                            best_insert_index_2 = j
    return best_chosen_route,best_insert_pickup,best_insert_index_1,best_insert_delivery,best_insert_index_2

def greedy_2_insertion(current,random_state):
    inserted = list(np.concatenate(list(current.routes.values())).flat)
    pool = list(set(task_pool)-set(inserted))
    print('greedy_2_pool', pool)
    while len(pool) > 0:
        chosen_route, pick_node, insert_index_1, delivery_node, insert_index_2 = findRegretInsert(current,pool)
        current.routes[chosen_route].insert(insert_index_1,pick_node)
        current.routes[chosen_route].insert(insert_index_2,delivery_node)
        pool.remove(pick_node)
        pool.remove(delivery_node)
    return current

def findRegretInsert(current,pool):
    regret_n = 2
    best_insert_pickup = None
    best_insert_delivery = None
    best_chosen_route = None
    best_insert_index_1 = None
    best_insert_index_2 = None
    best_insert_cost = -float('inf')
    for i in range(0,len(pool),2):  # all p+d pair try all possible routes and all posible positions in that chosen route
        n_insert_cost = np.zeros(6) # make up one row, in order to hstack other rows
        for key,route in current.routes.items():  # loop over all routes
            for k in range(len(route)+1):
                for j in (k + 1, len(route)+2):
                    # temp = current.routes[key].copy()  # use to record only one change of the solution
                    temp_route = copy.deepcopy(route)
                    temp_route.insert(k, pool[i])
                    temp_route.insert(j, pool[i + 1])
                    if feasibility_control(key, temp_route): # check the inserted route feasibility, if Ture, record the result
                        distance, distance_vector, duration, delay, delay_vector = distance_time_estimation(key,temp_route)
                        new_cost = alpa * distance + beta * delay
                        cost = np.array([pool[i], pool[i + 1],key,k,j,new_cost])
                        n_insert_cost = np.vstack((n_insert_cost,cost))
        n_insert_cost = np.delete(n_insert_cost, 0, 0)  # delete the first make-up row
        n_insert_cost = n_insert_cost[n_insert_cost[:,5].argsort()]# sort the cost array by the Objective function: new cost
        delta_f = 0
        for r in range(1,regret_n): # r = 1 # regret step = 1
            delta_f = delta_f + n_insert_cost[r,5]-n_insert_cost[0,5] ## second best-first best
        if delta_f > best_insert_cost:  # find the regret cost
            best_chosen_route = n_insert_cost[0,2]  # route id      #[pool[i], pool[i + 1],route_id, k , j , new_cost]
            best_insert_pickup = n_insert_cost[0,0] # pool[i],pick-up point
            best_insert_delivery = n_insert_cost[0,1] # pool[i+1],delivery point
            best_insert_index_1 = n_insert_cost[0,3] # index k, provider insertion position
            best_insert_index_2 = n_insert_cost[0,4] #index j, customer insertion position
            best_insert_cost = delta_f
    return int(best_chosen_route), int(best_insert_pickup), int(best_insert_index_1), int(best_insert_delivery), int(best_insert_index_2)

# def fast_greedy_insertion(current,random_state): ## back up.
#     return current

# generate solution state
# initial Route object


key = driver['unique_id'].tolist()
values =[[] for _ in range(len(key))]
routes_initial = dict(zip(key,values))
solution = PDPState(routes_initial) # empty solution

task_pool = task_map['id'].tolist()
driver_list = driver['unique_id'].tolist()

def nearest_driver(driver_list,task_id): #driver_list: id of available drivers
    distance = []
    for driver_id in driver_list:
        distance.append(nx.shortest_path_length(G,source=driver.loc[driver_id,'id'],target= task_map.loc[task_id,'point of interest'],weight='length'))
    print(distance)
    chosen_driver_id = driver_list[distance.index(min(distance))]
    return chosen_driver_id

# control if the position swap is valid or not.
def swap_control(route,first_pos,second_pos):
    route[first_pos],route[second_pos] = route[second_pos],route[first_pos]
    for i in range(len(route)):
        if route[i]%2 ==0: # pick_up point
            if route.index(route[i])> route.index(route[i]+1): # pick_up is after delivery, false
                return False
            else:
                pass
        else: # delivery point
            if route.index(route[i])< route.index(route[i]-1): # delivery is before pick_up,false
                return False
            else:
                pass
    return True

# call HC algorithm to improve the route
def Hillclimbing(unique_id,route):
    best_route = route
    for i in range(len(route)):
        for j in range(i+1,len(route)):
            can_route = copy.deepcopy(route) # give it the original route to do the swap
            if swap_control(can_route,i,j):
                print('can_route', can_route)
                if route_cost(unique_id, can_route) < route_cost(unique_id,best_route):  # 更改过后,candidate is better than best
                    best_route = can_route
                else:
                    pass
            else:
                pass
    return best_route

# prepare initial solution with HC algorithm
def sequential_insertion(task_pool,solution):
    while len(task_pool)>0:
        chosen_driver_id = nearest_driver(driver_list, task_pool[0])
        driver_list.remove(chosen_driver_id)# update remaining driver
        solution.routes[chosen_driver_id].append(task_pool[0])
        solution.routes[chosen_driver_id].append(task_pool[1])# insert the first order in the pool
        task_pool.remove(task_pool[0])
        task_pool.remove(task_pool[0])# find the nearest driver
        insert_list = []
        no_improvement = 0
        for i in range(0,len(task_pool),2): # for the remaining orders,try to insert one by one
            solution.routes[chosen_driver_id].append(task_pool[i])
            solution.routes[chosen_driver_id].append(task_pool[i+1]) ##[0,1,2,3]
            best_route = Hillclimbing(chosen_driver_id,solution.routes[chosen_driver_id])
            print('final best route',best_route)
            if feasibility_control(chosen_driver_id,best_route):
                insert_list.append(task_pool[i])
                insert_list.append(task_pool[i+1])
            else:
                solution.routes[chosen_driver_id].remove(task_pool[i])
                solution.routes[chosen_driver_id].remove(task_pool[i + 1])
                no_improvement +=1
            if no_improvement == 10: ## some route has the overduration problem, use this to avoid unnecessary trial
                solution.routes[chosen_driver_id] = Hillclimbing(chosen_driver_id, solution.routes[chosen_driver_id]) ## final update the best route
                break
        task_pool = [x for x in task_pool if x not in insert_list]
        print(task_pool)
    return solution

initial_solution = sequential_insertion(task_pool,solution)

print(initial_solution.routes)
initial_objective = initial_solution.objective()
inital_distance,initial_delay = initial_solution.distance_delay()

print('inital_distance,initial_delay,initial_objective',inital_distance,initial_delay,initial_objective)

SEED = 9876
random_state = rnd.RandomState(SEED)
alns= ALNS(random_state)
print(alns._rnd_state)
alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(longest_segment_removal)
alns.add_destroy_operator(path_removal)

alns.add_repair_operator(random_repair)
alns.add_repair_operator(greedy_insertion)
alns.add_repair_operator(greedy_2_insertion)

criterion = SimulatedAnnealing(50,0.95,1)
result = alns.iterate(initial_solution,[3,2,1,0.5],0.8,criterion,iterations=100,collect_stats=True)

solution = result.best_state
best_objective = solution.objective()
best_distance,best_delay = solution.distance_delay()

print('best_distance,best_delay,best_objective',best_distance,best_delay,best_objective)
with open('50task_best_solution.csv','w') as csvfile:
     writer = csv.writer(csvfile)
     for key,value in solution.routes.items():
         writer.writerow([key,value])
csvfile.close()


print('Best heuristic objective is {0}.'.format(best_objective))
print('This is {0:.1f}% better than the initial solution, which is {1}.'
      .format(100 * (initial_objective-best_objective) / initial_objective, initial_objective))

_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax)

figure = plt.figure("operator_counts", figsize=(14, 6))
figure.subplots_adjust(bottom=0.15, hspace=.5)
result.plot_operator_counts(figure=figure, title="Operator diagnostics", legend=["Best", "Better", "Accepted"])
plt.show()










