import heapq as hq
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
import numpy as np

#***************INITIALIZE OF CONSTANTS***************
#*variables to calculate time taken
start_time = None
end_time = None
#*dimensional constants
factor_dist_between_lines = np.sqrt(1**2 + (75/130)**2 ) #for hexagon perimeter
slope_hexagon = 15/26 #for hexagon lines
radius_goal = 1.5 #zone of acceptance of goal
width_space = 1200 #horizontal dimension of space
height_space = 500 #vertical dimension of space
th_distance = 0.5 #threshold for distance difference between nodes
th_angle = 30 #threshold for angle
normal_x_vector = (1, 0) #vector for orientation of the robot's front-facing
#*define duplicate check matrix and action set constant
factor_distance = 1 / th_distance
factor_angle = 1 / th_angle
rows_check_space = int(width_space * factor_distance)
cols_check_space = int(height_space * factor_distance)
angle_check_space = int(360 * factor_angle)
angle_values_raw = list(range(0, 7)) + list(range(-5, 0))
angle_values_real = th_angle * np.array(angle_values_raw)
transformation_matrix = [ [0,-1,cols_check_space],[1,0,0],[0,0,1] ] #[[scale_factor, 0], [0, scale_factor]]
action_operator = {
    '60_up': 2,
    '30_up': 1,
    '0': 0,
    '30_down': -1,
    '60_down': -2,
}
action_set = tuple(action_operator.keys())


#***************USER INPUT FUNCTIONS***************
def coordinate_input(name):
    value_provided = False
    while not value_provided:
        print(f'Enter coordinates of the { name } state:\n')
        try:
            input_coord = input(
                f'Provide horizontal,vertical position and orientation in multiples of {th_angle} \n separated by comma ( eg: 3,8,-60 ).')
            # print(input_coord)
            coord_process = input_coord.split(',')
            coord_process = [ int(element) for element in coord_process ]
            # user provided more or less elements than allowed
            if len(coord_process) != 3:
                raise Exception('NotExactElements')
            # coordinate is not valid
            angle_incorrect =  coord_process[2] % th_angle !=0
            # print(coord_process)
            # print(angle_incorrect)
            if angle_incorrect:
                raise Exception('CoordsNotValid')
            confirm = input('Confirm coordinate? (y/n): ')
            if confirm == 'y':
                print(f'The coordinate is: { coord_process }')
                return (coord_process[0], coord_process[1], coord_process[2])
            else:
                print('*****Must write coordinate again****')
                raise Exception('Repeat')
        except ValueError as error:
            print(error)
            print(
                f'Invalid input for the coordinate. Could not convert to integer all values.')
        except Exception as err:
            args = err.args
            if 'NotExactElements' in args:
                print('im here')
                print('Coordinate should have exactly two values. Please try again.')
            elif 'CoordsNotValid' in args:
                print('angle not valid. Please try again.')
            else:
                print(err)

def param_robot_input():
    value_provided = False
    while not value_provided:
        print(f'Enter parameters for the robot:\n')
        try:
            input_clearance = input('Enter clearance of the obstacles in mm')
            clearance_robot = float(input_clearance)
            input_radius = input('Enter radius of the robot in mm:')
            radius_robot = float(input_radius)
            input_step_size = input(
                'Enter step size of the robot, between 1 and 10 units')
            step_size_robot = float(input_step_size)
            if not (step_size_robot >= 1 and step_size_robot <= 10):
                raise Exception('NotInRange')
            confirm = input('Confirm params? (y/n):')
            if confirm == 'y':
                params_robot = (clearance_robot, radius_robot, step_size_robot)
                print(f'robot parameters are {params_robot[0]} clearance, {params_robot[1]} radius, {params_robot[2]} step')
                return params_robot
            else:
                print('*****Must write parameters again****')
                raise Exception('Repeat')
        except ValueError as error:
            print(error)
            print(
                f'Invalid input for the parameter. Could not convert to integer all values.')
        except Exception as err:
            args = err.args
            if 'NotInRange' in args:
                print(f'Step size not in range. Please write again all parameters')
            else:
                print(err)

#***************CHECK FUNCTIONS***************
def check_in_obstacle(state, border):
	tl = border
	x_pos, y_pos = state
	in_obstacle = np.zeros(6, dtype=bool)
	#outside of space
	if x_pos < 0 or y_pos < 0:
		#print(f'outside of space')
		return True
	if x_pos > width_space or y_pos > height_space:
		#print(f'outside of space')
		return True
	#first obstacle
	in_obstacle[0] = ( x_pos >= 100-tl and x_pos <= 175+tl ) and (y_pos >= 100-tl and y_pos <= height_space)
	if in_obstacle[0]:
		#print(f'first obstacle rectangle detected')
		return True
	#second obstacle
	in_obstacle[1] = ( x_pos >= 275-tl and x_pos <= 350+tl ) and (y_pos >= 0 and y_pos <= 400+tl )
	if in_obstacle[1]:
		#print(f'second obstacle rectangle detected')
		return True
	#third obstacle
	half_primitive = np.zeros(5, dtype=bool)
	half_primitive[0] = ( y_pos + slope_hexagon*x_pos - 475 + ( tl * factor_dist_between_lines) ) >= 0
	half_primitive[1] = ( y_pos + slope_hexagon*x_pos - 475 - ( ( (2*tl) + 260 ) * factor_dist_between_lines) ) <= 0
	half_primitive[2] = ( y_pos - slope_hexagon*x_pos + 275 + ( tl * factor_dist_between_lines) )  >= 0
	half_primitive[3] = ( y_pos - slope_hexagon*x_pos + 275 - ( ( (2*tl) + 260 ) * factor_dist_between_lines) )<= 0
	half_primitive[4] = x_pos >= 520-tl and x_pos <= 780+tl
	in_obstacle[2] = half_primitive.all()
	if in_obstacle[2]:
		#print(f'third obstacle hexagon detected')
		return True
	#fourth obstacle
	polygon_1 = np.zeros(3, dtype=bool)
	polygon_1[0] = ( x_pos >= 900-tl and x_pos <= 1100+tl ) and ( y_pos >= 375-tl and y_pos <= 450+tl )
	polygon_1[1] = ( x_pos >= 1020-tl and x_pos <= 1100+tl ) and ( y_pos >= 125 and y_pos <= 375 )
	polygon_1[2] =  ( x_pos >= 900-tl and x_pos <= 1100+tl ) and (y_pos >= 50-tl and y_pos <= 125+tl  )
	in_obstacle[3] = any(polygon_1)
	if in_obstacle[3]:
		#print(f'fourth obstacle ] shape detected')
		return True
	#border wall 1
	polygon_2 = np.zeros(3, dtype=bool)
	polygon_2[0] = ( x_pos >= 0 and x_pos <= 100-tl ) and (y_pos >= height_space-tl and y_pos <= height_space)
	polygon_2[1] = ( x_pos >= 0 and x_pos <= 5 ) and (y_pos >= tl and y_pos <= height_space-tl)
	polygon_2[2] =  ( x_pos >= 0 and x_pos <= 275-tl ) and (y_pos >= 0 and  y_pos <= tl )
	in_obstacle[4] = any(polygon_2)
	if in_obstacle[4]:
		#print(f'walls left detected')
		return True
	#border wall 2
	polygon_3 = np.zeros(3, dtype=bool)
	polygon_3[0] = ( x_pos >= 175+tl and x_pos <= width_space ) and (y_pos >= height_space-tl and y_pos <= height_space)
	polygon_3[1] = ( x_pos >= width_space-tl and x_pos <= width_space ) and ( y_pos >= tl and y_pos <= height_space-tl)
	polygon_3[2] =  ( x_pos >= 350+tl and x_pos <= width_space ) and ( y_pos >= 0 and y_pos <= tl )
	in_obstacle[5] = any(polygon_3)
	if in_obstacle[5]:
		#print(f'walls right detected')
		return True
	return False

#FUNCTIONS FOR GENERAL VARIABLES
def rotation_vectors_by(angle):
    if angle < 0:
        angle = angle % 360
    angle_rad = np.radians(angle)
    return np.array([[round(np.cos(angle_rad), 2),
                        round(-np.sin(angle_rad), 2)],
                            [round(np.sin(angle_rad), 2),
                                round(np.cos(angle_rad), 2)]])
#FUNCTIONS FOR APPLY A* ALGORITHM
def round_float(number):
    if number % 1 < 0.25:
        return int(number)
    elif number % 1 < 0.75:
        return int(number) + 0.5
    else:
        return int(number) + 1

def get_vector(node_a, node_b):
    return tuple(x - y for x, y in zip(node_a, node_b))

def distance(node_a, node_b):
    substract_vector = get_vector(node_a, node_b)
    #? Euclidean distance squared has given better performance
    return substract_vector[0]**2 + substract_vector[1]**2

def apply_action(state, type_action):
    x_pos, y_pos, theta = state
    action_to_do = action_operator.get(type_action, None)
    if action_to_do is None:
        return None
    value_angle = angle_values_real[action_to_do]
    rotation_index = int(theta * factor_angle)
    vector_front = np.dot(rotation_angle_matrices[rotation_index], normal_x_vector)
    new_vector = np.dot(rotation_angle_matrices[action_to_do], vector_front) * step_size
    x_pos_new = round_float(round(x_pos + new_vector[0], 2))
    y_pos_new = round_float(round(y_pos + new_vector[1], 2))
    angle_degrees = theta + value_angle
    if angle_degrees > 180:
        angle_degrees = angle_degrees - 360
    if angle_degrees < -180:
        angle_degrees = angle_degrees + 360
    return (x_pos_new, y_pos_new, angle_degrees)

def convert_check_matrix(node):
    x_pos, y_pos, theta = node
    x_pos_true = int(x_pos * factor_distance)
    y_pos_true = int(y_pos * factor_distance)
    angle_index = int(theta * factor_angle)
    return (x_pos_true, y_pos_true, angle_index)

def add_to_check_matrix(node):
    x_pos_true, y_pos_true, angle_index = convert_check_matrix(node)
    check_duplicates_space[x_pos_true, y_pos_true, angle_index] = 1

def is_duplicate(node):
    x_pos_true, y_pos_true, angle_index = convert_check_matrix(node)
    if check_duplicates_space[x_pos_true, y_pos_true, angle_index] == 1:
        return True
    return False

def action_move(current_node, action):
    """
    Args:
        current_node (Node): Node to move

    Returns:
        Node: new Node with new configuration and state
    """
    state_moved = apply_action(current_node[5:], action)
    # *check by the state duplicate values between the children
    node_already_visited = is_duplicate(state_moved)
    if node_already_visited:
        return None
    # *check new node is in obstacle space
    if check_in_obstacle(state_moved[0:2], border_obstacle):
        return None
    new_cost_to_come = current_node[1] + step_size
    new_cost_to_go = distance(state_moved[0:2], goal_state[0:2]) #heuristic function
    new_total_cost =  new_cost_to_come + new_cost_to_go
    new_node = (new_total_cost, new_cost_to_come, new_cost_to_go) + (-1, current_node[3]) + state_moved
    return new_node

def check_goal_reached(node_a, goal):
    dist_centers = distance(node_a[0:2], goal[0:2])
    orientation_valid = np.abs(node_a[2] - goal[2]) <= 2*th_angle
    center_robot_in_radius_goal = dist_centers < radius_goal**2
    if center_robot_in_radius_goal and orientation_valid:
        return True
    return False



#USER VARIABLES
initial_state = coordinate_input('initial')
goal_state = coordinate_input('goal')
clearance, radius_robot, step_size = param_robot_input()
border_obstacle = clearance + radius_robot
verify_initial_position = check_in_obstacle(initial_state[0:2], border_obstacle)
verify_goal_position = check_in_obstacle(goal_state[0:2], border_obstacle)
if verify_initial_position:
    print("START HITS OBSTACLE!! Please run the program again.")
    exit(0)
if verify_goal_position:
    print("GOAL HITS OBSTACLE!! Please run the program again.")
    exit(0)

#GENERAL VARIABLES FOR A*
generated_nodes = []  # open list
generated_nodes_total = []  # for animation of all

#* Set class has given excellent performance for Repeated-state checking
visited_nodes = []  # full list node visited
visited_vectors = {} # for animation
goal_path = np.array([]) #nodes shortest path
hq.heapify(generated_nodes)

check_duplicates_space = np.zeros(
    (rows_check_space, cols_check_space, angle_check_space))
#* Provides matrices of rotation to instead apply sin/cos
rotation_angle_matrices = np.array([ rotation_vectors_by(angle) for angle in angle_values_real ])

