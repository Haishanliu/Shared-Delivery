import random

from alns import ALNS,State
from alns.criteria import SimulatedAnnealing

import copy
import itertools
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy as np

class order(object): ## collect the order information
    def __init__(self,order_id,order_request_type,order_EPT,order_ETA,restaurant_location,customer_location,pickup_service_time,delivery_service_time):
        self.order_id = order_id
        self.order_request_type = order_request_type
        self.order_EPT = order_EPT
        self.order_ETA = order_ETA
        self.restaurant_location = restaurant_location
        self.customer_location  = customer_location
        self.pickup_service_time = pickup_service_time
        self.delivery_service_time = delivery_service_time

max_order_number = 10 ##每次最多有10单 （p+d）
class driver_route(object): ## generate route for driver, collect driver's schedule and location
    def __init__(self,driver_id,route_start_time,route_end_time,driver_location):
        self.driver_id = driver_id
        self.route_start_time = route_start_time
        self.route_end_time = route_end_time
        self.driver_location = driver_location
        self.order_sequence= np.arrary([])# ([order_id, request_type],[order_id, request_type],[order_id, request_type])

    # insert an order to the route
    def insert_order_to_route(self,index,order):
        np.insert(self.order_sequence,index,order)

    # remove an order from the route，give a position
    def remove_order_from_route(self,index): ## remove pick_up and delivery point at the same time
        order_id = self.order_sequence[index].order_id
        new_index = np.argwhere(self.order_sequence.order_id == order_id )
        if len(new_index)==2:  # only remove the order with p+d pair. If not, cannot remove a sigle p or d
           order_to_remove = self.order_sequence[new_index]
           np.delete(self.order_sequence,new_index)
           sort_array = order_to_remove[np.argsort(order_to_remove[:,1])]
           pick_up = sort_array[1]
           delivery = sort_array[0]
           return pick_up,delivery
        else:
            return None

    ## determine distance of one route
    def distance(self): ## input: a route; output: the total travel distance and distace between two orders
        distance = 0
        distance_vector = []
        if len(self.order_sequence.shape[0])==0:
            print('this route has 0 element')
        distance_vector = distance_vector.append(shortest_path(self.driver_location, self.order_sequence[0])) #driver to first order place
        distance += shortest_path(self.driver_location, self.order_sequence[i])
        for i in range(self.order_sequence.shape[0] - 1):
            distance_vector = distance_vector.append(shortest_path(self.order_sequence[i],self.order_sequence[i+1]))
            distance += shortest_path(self.order_sequence[i],self.order_sequence[i+1] )
        return distance, distance_vector

    ## determine arrival time of each point.
    def delay(self):
        distance = 0
        time = self.route_start_time
        time = time + shortest_path(self.driver_location, self.order_sequence[0]) / v  ##from driver location to first place
        delay = []
        if (self.order_sequence[0].order_request_type == 'p'):
            if time < self.order_sequence[0].EPT:
                  time = self.order_sequence[0].EPT
        else:
            if (time > self.order_sequence[0].ETA):
                delay.append(time - self.order_sequence[0].ETA)
        for i in range(self.order_sequence.shape[0] - 1):
            time += shortest_path(self.order_sequence[i], self.order_sequence[i + 1]) / v
            if (self.order_sequence[i + 1].order_request_type == 'p'):
                if (time < self.order_sequence[i + 1].EPT):
                    time += self.order_sequence[i + 1].EPT
                time += 1  ##服务时间 1min
            else:
                if (time > self.order_sequence[i + 1].ETA):
                    delay.append(time - self.order_sequence[i + 1].ETA)
                time += 1  ##服务时间 1min
        duration = time - self.route_start_time
        return duration,sum(delay),delay ## return duration of a route


    ## feasibility control #控制前后，控制由同一辆车送一个订单，控制订单数，控制司机的schedule，控制
    def feasibility_control(self): ## 给入一个 route，判断是否可行
        feasibility_key = True
        duration = self.delay()[0]
        if duration > (self.route_end_time-self.route_start_time):
            return False
        if self.order_sequence.shape[0]>=max_order_number:
            return False

alpa = 0.3
beta = 0.7 ## weights for objective function
class PDPState(State): ## given the orders and generated routes, what is the objective value?
   def __init__(self,routes):
       self.routes = routes

   def order_number(self):
       total_order = 0
       for route in self.routes:
           total_order += route.order_sequence.shape[0]
       return total_order

   def copy(self):
        return copy.deepcopy(self)

   def objective(self):
        distance_total = 0
        delay_total = 0
        #duration = 0
        for route in self.routes:
            duration, delay, distance = route.estimation()
            delay_total += delay
            distance_total += distance
        return alpa*distance_total+beta*delay_total

    # time waste  route arrival time of the pickup location - EPT
    # time delay. route arrival time of the delivery location - ETA
    # energy consumption: sum of energy consumption of two consecutive points.


# define destroy operator
degree_of_destruction = 0.2
def orders_to_remove(state):
    return int(len(state.order_number())*degree_of_destruction)

## removal operators##

def random_removal(current,random_state): ##routes nd arrary of route.[[[order1],[order2],[order3]],[],[]]
    pool = []
    destroyed = current.copy()
    for i in range(orders_to_remove(destroyed)):
        index_1 = random.randint(0,destroyed.routes.shape[0])## randomly pick a route
        if destroyed.routes[index_1].order_sequence.shape[0]==0:
            index_1 = random.randint(0, destroyed.routes.shape[0]-1)
        index_2 = random.randint(0, destroyed.routes[index_1].order_sequence.shape[0]-1)# pick a order p or d in the route. Then this p or d should has a corresponding d or p in the route
        order_1,order_2 = destroyed.routes[index_1].remove_order_from_route(index_2)
        pool.append(order_1)
        pool.append(order_2)# remove the pickup and delivery request at the same time
    return destroyed,pool

def longest_segment_removal(current,random_state): # remove the top_N longest segments between two orders
    pool = []
    destroyed = current.copy()
    for i in range(orders_to_remove(destroyed)):
        average_route_distance = []
        for route in destroyed.routes:  # find the worst route  distance/n # How to cope with n ==0
            distance,distance_vector = route.distance()
            if len(route)>0: # exclude the empty route
               average_route_distance.append(distance/len(route))
        index_1 = average_route_distance.index(max(average_route_distance))
        distance, distance_vector = destroyed.routes[index_1].distance()  # find the worst order in the route
        index_2 = distance_vector.index(max(distance_vector))
        order_1, order_2 = destroyed.routes[index_1].remove_order_from_route(index_2)
        pool.append(order_1)
        pool.append(order_2)
    return destroyed,pool

def path_removal(current,random_state): ## remove the longest distance path
    destroyed = current.copy()
    route_distance = []
    for route in destroyed.routes:
        distance, distance_vector = route.distance()
        route_distance.append(distance)
    index_1 = route_distance.index(max(route_distance))
    pool = destroyed.routes[index_1]
    destroyed.route[index_1] = [[]]
    return destroyed,pool

def longest_delay_removal(current,random_state):
    pool = []
    destroyed = current.copy()

def shaw_removal(current,random_state):
    destroyed = current.copy()


    return destroyed


## insertion operators ##
def random_repair(current,pool): # pool=[[3,'p'],[3,'d'],[5,'p'],[5,'d'],...]
    for i in range(len(pool)):
        chosen_route = random.choice(current.routes) # choose a route
        k = random.randint(0,len(chosen_route)) # randomly choose a position
        chosen_route.insert_order_to_route(k, pool[i]) # insert provider
        j = random.randint(k+1,len(chosen_route))  # randomly choose a position
        chosen_route.insert_order_to_route(j, pool[i+1]) # insert customer
        i = i+2
    pool=[]
    return current

def greedy_insertion(current,pool,random_state):
    while len(pool)>0:
        chosen_route,pick_node,insert_index_1,delivery_node,insert_index_2 = findGreedyInsert(current,pool)
        current.routes[chosen_route].insert_order_to_route(pick_node,insert_index_1)
        current.routes[chosen_route].insert_order_to_route(delivery_node, insert_index_2)
        pool.remove(pick_node)
        pool.remove(delivery_node)
    return current

def findGreedyInsert(current,pool): # find best orders to be inserted in best position
    best_insert_pickup = None
    best_insert_delivery = None
    best_chosen_route = None
    best_insert_index_1 = None
    best_insert_index_2 = None
    best_insert_cost = float('inf')
    current_cost = current.ojective()
    while len(pool)>0:
        for i in range(len(pool)):# all p+d pair try all possible routes and all posible positions in that chosen route
            for route_id in range(len(current.routes)): # loop over all routes
                for k in range(len(current.routes[route_id])+1): # +1, cause it can insert at the end of the route.
                    for j in (k+1,len(current.routes[route_id])+1): # +1, cause it can insert at the end of the route.
                        temp = current.copy() # use to record only one change of the solution
                        temp.routes[route_id].insert_order_to_route(k,pool[i])
                        temp.routes[route_id].insert_order_to_route(j, pool[i+1])
                        if temp.routes[route_id].feasibility_control()==True : # check feasibility fisrt, then calculate objective
                           new_cost = temp.ojective()
                           deta_f = new_cost-current_cost # cost increase after insertion
                           if deta_f<best_insert_cost : # compare with best increase so far, use to pick the best order, and record the minimum postion
                              best_chosen_route = route_id
                              best_insert_pickup = pool[i]
                              best_insert_delivery = pool[i+1]
                              best_insert_index_1 = k
                              best_insert_index_2 = j
            i = i+2
    return best_chosen_route,best_insert_pickup,best_insert_index_1,best_insert_delivery,best_insert_index_2

def greedy_2_insertion(current,pool,random_state):
    while len(pool) > 0:
        chosen_route, pick_node, insert_index_1, delivery_node, insert_index_2 = findRegretInsert(current,pool)
        current.routes[chosen_route].insert_order_to_route(pick_node, insert_index_1)
        current.routes[chosen_route].insert_order_to_route(delivery_node, insert_index_2)
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
    current_cost = current.ojective()
    while len(pool) > 0:
        for i in range(len(pool)):  # all p+d pair try all possible routes and all posible positions in that chosen route
            n_insert_cost = np.zeros(1,6) # make up one row, in order to hstack other rows
            for route_id in range(len(current.routes)):  # loop over all routes
                for k in range(len(current.routes[route_id])+1):
                    for j in (k + 1, len(current.routes[route_id])+1):
                        temp = current.copy()  # use to record only one change of the solution
                        temp.routes[route_id].insert_order_to_route(k, pool[i])
                        temp.routes[route_id].insert_order_to_route(j, pool[i + 1])
                        if temp.routes[route_id].feasibility_control() == True: # check the inserted route feasibility, if Ture, record the result
                           new_cost = temp.ojective()
                           cost = np.array([pool[i], pool[i + 1],route_id,k,j,new_cost])
                           n_insert_cost = np.hstack(n_insert_cost,cost)
            n_insert_cost = np.delete(n_insert_cost, 0, 0)  # delete the first make-up row
            n_insert_cost = n_insert_cost[n_insert_cost[:,5].argsort()]# sort the cost array by the Objective function: new cost
            deta_f = 0
            for i in range(1,regret_n): # i = 1
                deta_f = deta_f + n_insert_cost[i,5]-n_insert_cost[0:,5] ## second best-first best
            if deta_f > best_insert_cost:  # after the insertion, the route should be feasible
                best_chosen_route = n_insert_cost[0,2]  # route id      #[pool[i], pool[i + 1],route_id, k , j , new_cost]
                best_insert_pickup = n_insert_cost[0,0] # pool[i],pick-up point
                best_insert_delivery = n_insert_cost[0,1] # pool[i+1],delivery point
                best_insert_index_1 = n_insert_cost[0,3] # index k, provider insertion position
                best_insert_index_2 = n_insert_cost[0,4] #index j, customer insertion position
                best_insert_cost = deta_f
            i = i + 2
    return best_chosen_route, best_insert_pickup, best_insert_index_1, best_insert_delivery, best_insert_index_2

# def fast_greedy_insertion(current,random_state): ## back up.
#     return current
