import csv
import random
import networkx as nx
import os
import time
import pandas as pd
import numpy as np
import numpy.random as rnd
import copy
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        self.best_sol = None
        self.r1 = 30
        self.r2 = 18
        self.r3 = 12
        self.rho = 0.6 # reaction factor, how quickly it react to the effectiveness. rho =1, only count for last upate.
        self.d_weight = np.ones(3) # 3 removal operators
        self.d_select = np.zeros(3)
        self.d_score = np.zeros(3)
        self.d_history_select = np.zeros(3)
        self.d_history_score = np.zeros(3)
        self.r_weight = np.ones(4)  # 3 removal operators
        self.r_select = np.zeros(4)
        self.r_score = np.zeros(4)
        self.r_history_select = np.zeros(4)
        self.r_history_score = np.zeros(4)
        self.distance_matrix = {}
        self.time_matrix= {}
        self.task_dict = {} # key: task id, value: task_object
        self.task_id_list = []
        self.driver_dict = {}
        self.driver_id_list = []

class Task():
    def __init__(self):
        self.id =0
        self.location_id = 0 # point of interest
        self.EPT = None # if pick_up, record EPT, if delivery, record ETA
        self.ETA = None
        self.waiting_time = None
        self.delay = None # if pick up, record waiting time, if delivery, record delay. # every node record its own waiting time or delay
        self.x_coord = None
        self.y_coord = None


class Driver():
    def __init__(self):
        self.id = 0
        self.x_coord = None
        self.y_coord = None
        self.start_time = None
        self.end_time = None


class Sol(): ## given the orders and generated routes, what is the objective value? routes: array of route object

    def __init__(self,routes): # attribute
        self.routes = routes
        self.obj = None
        self.total_distance = None
        self.delay = None

    def order_number(self): #method
        total_order = 0
        for key in self.routes.keys(): # loop over the dictionary
            total_order += len(self.routes[key])/2 # 每个route 的task sequence， then divide by 2
        return total_order

    def copy(self):
        return copy.deepcopy(self)

def sol_objective(sol,model,alpa = 3.33,beta = 0.3): # calculate solution objective, total distance and total delay
    distance_total = 0
    delay_total = 0
    for key in sol.routes:
        distance,delay,duration,route_obj = distance_time_estimation(sol.routes[key],model)
        delay_total += delay
        distance_total += distance
    sol.obj = alpa*distance_total+beta*delay_total
    sol.total_distance = distance_total
    sol.delay = delay_total


def readCSVFile(routing_result_file,task_file,driver_file,model):

    with open(routing_result_file,'r') as f:
        routing_result = csv.DictReader(f)
        for row in routing_result:
            from_point = int(row['from'])
            to_point = int(row['to'])
            model.distance_matrix[from_point,to_point] = float(row['travel_distance'])/1000 # in kilometer
            model.time_matrix[from_point,to_point] = float(row['travel_time'])/60 # convert to min

    with open(task_file,'r') as f:
        task_reader = csv.DictReader(f)
        for row in task_reader:
            task = Task()
            task.id = int(row['id'])
            if int(row['id']) %2 ==0: # pick_up
                task.location_id = int(float(row['restaurant_id']))
                task.EPT = float(row['EPT'])
                task.x_coord = float(row['restaurant_Lat']) # use to estimated the first segment distance
                task.y_coord = float(row['restaurant_Lon'])
                model.task_dict[task.id] = task
                model.task_id_list.append(task.id)
            else: #delivery 不记录x_coord, y_coord
                task.location_id = int(float(row['customer_id']))
                task.ETA = float(row['ETA'])
                model.task_dict[task.id] = task
                model.task_id_list.append(task.id)
    with open(driver_file,'r') as f:
        driver_reader = csv.DictReader(f)
        for row in driver_reader:
            driver = Driver()
            driver.id = int(row['unique_id'])
            driver.x_coord = float(row['Lat'])
            driver.y_coord = float(row['Lon'])
            driver.start_time = float(row['start_time'])
            driver.end_time = float(row['end_time'])
            model.driver_dict[driver.id] = driver
            model.driver_id_list.append(driver.id)

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
task_map = pd.read_csv(r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\task_map.csv',index_col=False) # map of task and point of interest
driver = pd.read_csv(r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\driver_10_unique_id.csv') # driver information also assign unique id to each driver

def distance_time_of_2task(task_1_id,task_2_id,model): ## return travel_distance and travel time between any two points
    '''
    :param task_1_id: task_id, int
    :param task_2_id: task_id, int
    :param model:
    '''
    task_1 = model.task_dict[task_1_id] # use task_dict to map back to the exact task object
    task_2 = model.task_dict[task_2_id]
    distance = model.distance_matrix[task_1.location_id,task_2.location_id] # use distance_matrix to get the travel distance
    time = model.time_matrix[task_1.location_id,task_2.location_id]
    return distance,time

## determine distance of one route
def distance_time_estimation(task_sequence,model,alpa = 3.33, beta = 0.3): # only for one route
    '''
    :param task_sequence: given the task sequence, the driver starts working 10 min before the first EPT. And the driver is in the restaurant already
    '''
    distance = 0
    delay = 0
    duration = 0
    if len(task_sequence)==0: # empty route
        route_obj = 0
        return distance,delay,duration,route_obj
    else:
        start_time = model.task_dict[task_sequence[0]].EPT-10 # first task should be pick up
        current_time = start_time # 假设driver 在接第一个订单的前10分钟上班
        model.task_dict[task_sequence[0]].waiting_time = 0 #第一个order，直接取，driver到了直接取
        for i in range(len(task_sequence) - 1):
            travel_distance,travel_time = distance_time_of_2task(task_sequence[i],task_sequence[i+1],model)
            distance += travel_distance
            current_time += travel_time
            if task_sequence[i+1]%2 == 0: # pick_up point
                if current_time < model.task_dict[task_sequence[i+1]].EPT:
                    model.task_dict[task_sequence[i + 1]].waiting_time = model.task_dict[task_sequence[i + 1]].EPT - current_time #record waiting time
                    current_time = model.task_dict[task_sequence[i+1]].EPT # update current time
                current_time += 1  #service time 1min
            else: # delivery point
                if current_time > model.task_dict[task_sequence[i+1]].ETA:
                    model.task_dict[task_sequence[i + 1]].delay = current_time - model.task_dict[task_sequence[i+1]].ETA
                    delay +=  model.task_dict[task_sequence[i + 1]].delay
                else:
                    model.task_dict[task_sequence[i + 1]].delay = 0
                current_time += 1 ##service time 1min
            duration = current_time - start_time
        route_obj = alpa*(distance) + beta*delay
        return distance,delay,duration,route_obj

## feasibility control #控制前后，控制由同一辆车送一个订单，控制订单数，控制司机的schedule，控制


def check_delay(task_sequence,model):
    task_sequence = [task_id for task_id in task_sequence if task_id %2 != 0]
    for task_id in task_sequence:
        if model.task_dict[task_id].delay>60:
            return False
        else:
            return True


def feasibility_control(task_sequence,model,max_order_number=20): ## 给入一个 route，判断是否可行,
    if len(task_sequence)<=max_order_number:
        duration = distance_time_estimation(task_sequence,model)[2]
        if duration <=120:
            return True
    else:
        return False


# define destroy operator
degree_of_destruction = 0.1 # fraction of removal 参数
def orders_to_remove(state):
    return int(state.order_number()*degree_of_destruction)*2 # total task number to remove

# removal operators#

def random_removal(current):
    pool = []
    print('random_removal')
    destroyed = current.copy()
    for i in range(orders_to_remove(destroyed)):
        key_pool = []
        for key, item in destroyed.routes.items():
            if len(item) != 0:
                key_pool.append(key)
        index_1 = random.choice(key_pool)## randomly pick a route_key, these routes are non-empty
        task = random.choice(destroyed.routes[index_1])  # randomly chosen a task
        if task%2 ==0: # pick up point
            if task + 1 in destroyed.routes[index_1]:
                destroyed.routes[index_1].remove(task)
                destroyed.routes[index_1].remove(task + 1)
                pool.append(task)
                pool.append(task+1)
        else: # delivery point
            if task - 1 in destroyed.routes[index_1]:
                destroyed.routes[index_1].remove(task - 1)
                destroyed.routes[index_1].remove(task)
                pool.append(task-1)
                pool.append(task)
    return destroyed,pool


def worst_removal(current,model): # remove the top_N worst orders based on the objective value
    pool = [] # remove 之后，算新的obejective
    print('worst_removal')
    destroyed = current.copy()
    r_cost  = {}
    for key,item in destroyed.routes.items():
        r_cost[key]=distance_time_estimation(item,model)[3] # 记住每条路没修改之前的cost
    n_insert_cost = np.zeros(3)
    for key,task_sequence in destroyed.routes.items():
        for task in task_sequence:
            route = task_sequence[:] # not use deepcpoy, too slow
            if task%2 == 0: #pick_up point
                route.remove(task)
                route.remove(task+1)
                deta_f = r_cost[key] - distance_time_estimation(task_sequence,model)[3] # insertion cost of the order
                cost = np.array([key,task, deta_f])
                n_insert_cost = np.vstack((n_insert_cost, cost))  # 记住route_id, 记住order_id,记住 cost
            else:
                pass
    n_insert_cost = np.delete(n_insert_cost, 0, 0)  # delete the first make-up row
    n_insert_cost = n_insert_cost[n_insert_cost[:, 2].argsort()] # sort by insertion cost,increasing
    q = orders_to_remove(destroyed)
    p = 5 # power of random number q. From Shaw's result. 5-20 could get better result
    while q>0:
        y = random.uniform(0, 1) # randomly choose a number from [0,1)
        index = int(np.shape(n_insert_cost)[0]*pow(y,p))
        choosen_entery = n_insert_cost[index,:]
        pool.append(int(choosen_entery[1]))
        pool.append(int(choosen_entery[1]+1))
        destroyed.routes[choosen_entery[0]].remove(choosen_entery[1]) #delivery point
        destroyed.routes[choosen_entery[0]].remove(choosen_entery[1]+1) # pick up point
        n_insert_cost = np.delete(n_insert_cost, index, 0)
        q = q-1
    return destroyed,pool


def relateness(model,r1,r2,f1=9,f2=3): # two task id. f1,f2 weight for dis and time_diff # from Ropke paper
    order_1_pick_up = model.task_dict[r1]
    order_1_delivery = model.task_dict[r1+1]
    order_2_pick_up = model.task_dict[r2]
    order_2_delivery = model.task_dict[r2+1]

    r1_dis = model.distance_matrix[order_1_pick_up.location_id, order_1_delivery.location_id]
    r1_time_diff = order_1_delivery.ETA - order_1_pick_up.EPT
    r2_dis = model.distance_matrix[order_2_pick_up.location_id, order_2_delivery.location_id]
    r2_time_diff = order_2_delivery.ETA - order_2_pick_up.EPT
    relate = f1*(r1_dis+ r2_dis)+f2*(r1_time_diff+r2_time_diff)
    return relate


def shaw_removal(current,model):
    print('shaw_removal')
    p = 5 # power of random number
    destroyed = current.copy()
    pool = []
    key, item = random.choice([(name, value)
                                      for name, values in current.routes.items()
                                      for value in values]) # randomly choose to task and put it in the pool
    if item%2 ==0: #pick_up point
        pool.append(item)
        pool.append(item+1)
        destroyed.routes[key].remove(item)
        destroyed.routes[key].remove(item+1)
    else: #delivery
        pool.append(item-1)# pick_up point
        pool.append(item) #delivery point
        destroyed.routes[key].remove(item-1)
        destroyed.routes[key].remove(item)
    while len(pool)<orders_to_remove(current):
        r = random.choice(pool) # 随机选一个order
        n_relate = np.zeros(3)
        for key in destroyed.routes.keys():
            for task in destroyed.routes[key]:
                if task%2 ==0: #pick_up point, 第一个碰到的肯定是 pick_up point
                    if r% 2 ==0:
                        relate = relateness(model,r,task) # calculate relatedness
                    else:
                        relate = relateness(model,r-1,task)
                    # print('relate',relate)
                    cost = np.array([key, task, relate])
                    n_relate = np.vstack((n_relate, cost))
                else:
                    continue
        n_relate = np.delete(n_relate, 0, 0)  # delete the first make-up row
        n_relate = n_relate[n_relate[:, 2].argsort()]  # sort by relatedness 越小，相似性越高
        y = random.uniform(0, 1)
        index = int(np.shape(n_relate)[0] * (y**p))
        choosen_entry = n_relate[index, :]
        pool.append(int(choosen_entry[1])) # pick_up point
        pool.append(int(choosen_entry[1]+1)) # delivery point
        destroyed.routes[choosen_entry[0]].remove(choosen_entry[1])  # pick_up
        destroyed.routes[choosen_entry[0]].remove(choosen_entry[1] + 1)  # delivery
    return destroyed,pool


## insertion operators ##
def random_repair(current,pool,model):
    print('random_pool', pool)
    for i in range(0,len(pool),2):
        index_1 = random.choice([*current.routes.keys()]) # choose a route_key
        k = random.randint(0,len(current.routes[index_1])) # randomly choose a position
        current.routes[index_1].insert(k, pool[i]) # insert provider
        j = random.randint(k + 1, len(current.routes[index_1]))  # randomly choose a position
        current.routes[index_1].insert(j, pool[i+1]) # insert customer
    sol_objective(current,model)
    return current

def greedy_insertion(current,pool,model):
    print('greedy_pool',pool)
    r_cost = {}
    for key, item in current.routes.items():
        r_cost[key] = distance_time_estimation(item,model)[3]  # 记住每条路没修改之前的cost
    while len(pool)>0:
        chosen_route,pick_node,insert_index_1,delivery_node,insert_index_2 = findGreedyInsert(current,pool,r_cost,model) # find greedy p+d pair and its greedy position
        current.routes[chosen_route].insert(insert_index_1,pick_node)
        current.routes[chosen_route].insert(insert_index_2,delivery_node)
        r_cost[chosen_route] = distance_time_estimation(current.routes[chosen_route],model)[3]#update r_cost
        pool.remove(pick_node)
        pool.remove(delivery_node)
    sol_objective(current,model)
    return current

def findGreedyInsert(current,pool,r_cost,model): # find best orders to be inserted in best position
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
                    temp_route = route[:]# only copy that route, not the whole solution
                    temp_route.insert(k, pool[i])
                    temp_route.insert(j, pool[i + 1])
                    if feasibility_control(temp_route,model): # check feasibility firstt, then calculate objective
                        cost = distance_time_estimation(temp_route,model)[3] # after insertion
                        delta_cost = cost - r_cost[key] # cost change after insertion
                        if delta_cost < best_cost : # compare with best increase so far, use to pick the best order, and record the minimum postion
                            best_cost = delta_cost
                            best_chosen_route = key
                            best_insert_pickup = pool[i]
                            best_insert_delivery = pool[i+1]
                            best_insert_index_1 = k
                            best_insert_index_2 = j
    return best_chosen_route,best_insert_pickup,best_insert_index_1,best_insert_delivery,best_insert_index_2

def greedy_2_insertion(current,pool,model):
    print('greedy_2_pool', pool)
    r_cost = {}
    for key, item in current.routes.items():
        r_cost[key] = distance_time_estimation(item,model)[3]
    while len(pool) > 0:
        chosen_route, pick_node, insert_index_1, delivery_node, insert_index_2 = findRegretInsert(current,pool,r_cost,model)
        current.routes[chosen_route].insert(insert_index_1,pick_node)
        current.routes[chosen_route].insert(insert_index_2,delivery_node)
        r_cost[chosen_route] = distance_time_estimation(current.routes[chosen_route],model)[3]  # update r_cost
        pool.remove(pick_node)
        pool.remove(delivery_node)
    sol_objective(current,model)
    return current

def findRegretInsert(current,pool,r_cost,model):
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
                    temp_route = route[:]
                    temp_route.insert(k, pool[i])
                    temp_route.insert(j, pool[i + 1])
                    if feasibility_control(temp_route,model): # check the inserted route feasibility, if Ture, record the result
                        cost = distance_time_estimation(temp_route,model)[3]
                        delta_cost = cost - r_cost[key]
                        delta_cost = np.array([pool[i], pool[i + 1],key,k,j,delta_cost])
                        n_insert_cost = np.vstack((n_insert_cost,delta_cost))
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

def greedy_3_insertion(current,pool,model):
    print('greedy_3_pool', pool)
    r_cost = {}
    for key, item in current.routes.items():
        r_cost[key] = distance_time_estimation(item,model)[3]
    while len(pool) > 0:
        chosen_route, pick_node, insert_index_1, delivery_node, insert_index_2 = findRegretInsert_3(current,pool,r_cost,model)
        current.routes[chosen_route].insert(insert_index_1,pick_node)
        current.routes[chosen_route].insert(insert_index_2,delivery_node)
        r_cost[chosen_route] = distance_time_estimation(current.routes[chosen_route],model)[3]  # update r_cost
        pool.remove(pick_node)
        pool.remove(delivery_node)
    sol_objective(current,model)
    return current

def findRegretInsert_3(current,pool,r_cost,model):
    regret_n = 3
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
                    temp_route = route[:]
                    temp_route.insert(k, pool[i])
                    temp_route.insert(j, pool[i + 1])
                    if feasibility_control(temp_route,model): # check the inserted route feasibility, if Ture, record the result
                        cost = distance_time_estimation(temp_route,model)[3]
                        delta_cost = cost - r_cost[key]
                        delta_cost = np.array([pool[i], pool[i + 1],key,k,j,delta_cost])
                        n_insert_cost = np.vstack((n_insert_cost,delta_cost))
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




# initial Route object
def nearest_driver(driver_list,task_id): #driver_list: id of available drivers
    distance = []
    for driver_id in driver_list:
        distance.append(nx.shortest_path_length(G,source=driver.loc[driver_id,'id'],target= task_map.loc[task_id,'point of interest'],weight='length'))
    # print(distance)
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
def Hillclimbing(unique_id,route,model):
    best_route = route
    for i in range(len(route)):
        for j in range(i+1,len(route)):
            can_route = route[:] # give it the original route to do the swap
            if swap_control(can_route,i,j):
                # print('can_route', can_route)
                if distance_time_estimation(can_route,model)[3] < distance_time_estimation(best_route,model)[3]:  # 更改过后,candidate is better than best
                    best_route = can_route
                else:
                    pass
            else:
                pass
    return best_route

# prepare initial solution with HC algorithm
driver_list = driver['unique_id'].tolist()

def sequential_insertion(task_pool,solution,model):
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
            best_route = Hillclimbing(chosen_driver_id,solution.routes[chosen_driver_id],model)
            # print('final best route',best_route)
            if feasibility_control(best_route,model):
                insert_list.append(task_pool[i])
                insert_list.append(task_pool[i+1])
            else:
                solution.routes[chosen_driver_id].remove(task_pool[i])
                solution.routes[chosen_driver_id].remove(task_pool[i + 1])
                no_improvement +=1
            if no_improvement == 5: ## some route has the overduration problem, use this to avoid unnecessary trial
                solution.routes[chosen_driver_id] = Hillclimbing(chosen_driver_id, solution.routes[chosen_driver_id],model) ## final update the best route
                break
        task_pool = [x for x in task_pool if x not in insert_list]
        # print(task_pool)
    return solution


def selectDestroyRepair(model):
    d_weight = model.d_weight
    d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
    d_cumsumprob -= np.random.rand()
    destroy_id = list(d_cumsumprob > 0).index(True)

    r_weight = model.r_weight
    r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
    r_cumsumprob -= np.random.rand()
    repair_id = list(r_cumsumprob > 0).index(True)
    return destroy_id, repair_id

def doDestroy(destroy_id,current,model):
    if destroy_id == 0:
        destroyed,pool = random_removal(current)
    elif destroy_id ==1:
        destroyed, pool = worst_removal(current,model )
    else:
        destroyed, pool = shaw_removal(current,model)
    return destroyed,pool

def doRepair(repair_id,destroyed,pool,model):
    if repair_id ==0:
        new_sol = random_repair(destroyed,pool,model)
    elif repair_id ==1:
        new_sol = greedy_insertion(destroyed,pool,model)
    elif repair_id ==2:
        new_sol = greedy_2_insertion(destroyed,pool,model)
    else:
        new_sol = greedy_3_insertion(destroyed,pool,model)
    return new_sol

def resetScore(model):
    model.d_select = np.zeros(3)
    model.d_score = np.zeros(3)

    model.r_select = np.zeros(4)
    model.r_score = np.zeros(4)


def updateWeight(model):
    for i in range(model.d_weight.shape[0]):
        if model.d_select[i] > 0:
            model.d_weight[i] = model.d_weight[i] * (1 - model.rho) + model.rho * model.d_score[i] / model.d_select[i]
        else:
            model.d_weight[i] = model.d_weight[i] * (1 - model.rho)
    for i in range(model.r_weight.shape[0]):
        if model.r_select[i] > 0:
            model.r_weight[i] = model.r_weight[i] * (1 - model.rho) + model.rho * model.r_score[i] / model.r_select[i]
        else:
            model.r_weight[i] = model.r_weight[i] * (1 - model.rho)
    model.d_history_select = model.d_history_select + model.d_select
    model.d_history_score = model.d_history_score + model.d_score
    model.r_history_select = model.r_history_select + model.r_select
    model.r_history_score = model.r_history_score + model.r_score

def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()

def run(routing_result_file,task_file,driver_file,r1,r2,r3,rho,phi,epochs,pu): #还有其他参数 shaw removal f1,f2,p (power of the random number)
    '''
    :param destroyed_frac: destroyed fraction of each removal operators
    :param r1: score of best solution found
    :param r2: score of better solution found
    :param r3: score of soultion does not improve the current soultion,but got accepted
    :param rho: reaction factor of weight
    :param phi: temperature reduction factor of SA
    :param epochs: iterations
    :param pu: frenquency of weight adjustment
    :return:
    '''

    model = Model() #实例化一个model
    model.r1 = r1
    model.r2 = r2
    model.r3 = r3
    model.rho = rho
    readCSVFile(routing_result_file,task_file,driver_file,model)

    history_best_obj = []
    key = driver['unique_id'].tolist()
    values = [[] for _ in range(len(key))]
    routes_initial = dict(zip(key, values))
    sol = Sol(routes_initial)  # empty solution

    task_pool = task_map['id'].tolist()
    start_1 = time.time()
    sol = sequential_insertion(task_pool,sol,model)
    sol_objective(sol,model) # 记录初始解的 value
    ini_run_time = time.time()-start_1
    print('initial solution',sol.routes)
    history_best_obj.append(sol.obj)
    print('initial_distance,initial_delay,initial_objective',sol.total_distance,sol.delay,sol.obj)
    print('initial solution running time',ini_run_time)
    model.best_sol = copy.deepcopy(sol)
    start_2 = time.time()
    for ep in range(epochs):
        T = sol.obj * 0.2
        resetScore(model)
        for k in range(pu):
            destroy_id, repair_id = selectDestroyRepair(model)
            model.d_select[destroy_id] += 1
            model.r_select[repair_id] += 1
            destroyed,pool= doDestroy(destroy_id,sol,model)
            new_sol = doRepair(repair_id,destroyed,pool,model)
            if new_sol.obj < sol.obj:
                sol = copy.deepcopy(new_sol)
                if new_sol.obj < model.best_sol.obj:
                    model.best_sol = copy.deepcopy(new_sol)
                    model.d_score[destroy_id] += model.r1
                    model.r_score[repair_id] += model.r1
                else:
                    model.d_score[destroy_id] += model.r2
                    model.r_score[repair_id] += model.r2
            elif random.random() <= np.exp(-(new_sol.obj-sol.obj)/T):
                sol = copy.deepcopy(new_sol)
                model.d_score[destroy_id] += model.r3
                model.r_score[repair_id] += model.r3
            T = T * phi
            print("%s/%s:%s/%s， best obj: %s" % (ep, epochs, k, pu, model.best_sol.obj))
            history_best_obj.append(model.best_sol.obj)
        updateWeight(model)
    best_run_time = time.time()-start_2
    print('best_soluton',model.best_sol.routes)
    print('best_distance,delay,objective',model.best_sol.total_distance,model.best_sol.delay,model.best_sol.obj)
    print(' best_run_time', best_run_time)
    plotObj(history_best_obj)



if __name__=='__main__':
    routing_result_file = r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\routing_result.csv'
    task_file = r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\task.csv'
    driver_file = r'C:\Users\Haishan Liu\PycharmProjects\pythonProject\PDPTW\testcode\scenario_1\driver_10_unique_id.csv'
    run(routing_result_file,task_file,driver_file,r1=33,r2=15,r3=9,rho=0.8,phi=0.9,epochs=100,pu=100)















