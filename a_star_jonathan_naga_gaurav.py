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
	"""
	Asks the user to input the coordinates of a specific state.

	Args:
		name (str): The name of the state being inputted (e.g. "initial" or "goal").

	Returns:
		tuple: The horizontal, vertical, and orientation coordinates of the state.

	Raises:
		ValueError: If the input is not a valid coordinate.
		Exception: If the user does not confirm the input.

	"""
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
				print(f'The coordinate is: { coord_process }\n')
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
	"""
	Asks the user to input the parameters of the robot.

	Returns:
		tuple: The clearance, radius, and step size of the robot.

	Raises:
		ValueError: If the input is not a valid parameter.
		Exception: If the user does not confirm the input.

	"""
	value_provided = False
	while not value_provided:
		print(f'Enter parameters for the robot:\n')
		try:
			input_clearance = input('Enter clearance of the obstacles in mm:\n')
			clearance_robot = float(input_clearance)
			input_radius = input('Enter radius of the robot in mm:\n')
			radius_robot = float(input_radius)
			input_step_size = input(
				'Enter step size of the robot, between 1 and 10 units(mm):\n')
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
	"""
	This function checks if a given state is within the obstacle space.

	Args:
		state (tuple): The horizontal and vertical coordinates of the state.
		border (int): The clearance of the obstacles.

	Returns:
		bool: True if the state is within the obstacle space, False otherwise.

	"""
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
	"""
	Returns a 2x2 rotation matrix that rotates by the specified angle.

	Args:
		angle (float): The angle, in degrees, to rotate by.

	Returns:
		np.ndarray: A 2x2 rotation matrix.

	Raises:
		ValueError: If the angle is not a positive integer multiple of 360.

	"""
	if angle < 0:
		angle = angle % 360
	angle_rad = np.radians(angle)
	return np.array([[round(np.cos(angle_rad), 2),
						round(-np.sin(angle_rad), 2)],
							[round(np.sin(angle_rad), 2),
								round(np.cos(angle_rad), 2)]])
#FUNCTIONS FOR APPLY A* ALGORITHM
def round_float(number):
	"""
	This function rounds a float number to the nearest integer.

	Args:
		number (float): The float number to round.

	Returns:
		int: The nearest integer to the input float number.

	"""
	if number % 1 < 0.25:
		return int(number)
	elif number % 1 < 0.75:
		return int(number) + 0.5
	else:
		return int(number) + 1

def get_vector(node_a, node_b):
	"""
	This function returns the vector from node_a to node_b.

	Args:
		node_a (tuple): The first node.
	"""
	return tuple(x - y for x, y in zip(node_a, node_b))

def distance(node_a, node_b):
	"""
	Returns the Euclidean distance between two nodes.

	Args:
		node_a (tuple): The first node.
		node_b (tuple): The second node.

	Returns:
		float: The Euclidean distance between the two nodes.

	"""
	substract_vector = get_vector(node_a, node_b)
	#? Euclidean distance squared has given better performance
	return substract_vector[0]**2 + substract_vector[1]**2

def apply_action(state, type_action):
	"""
	Applies the given action to the given state.

	Args:
		state (tuple): The current state of the robot, as a tuple of its x and y coordinates and its orientation.
		type_action (str): The type of action to apply, as a string.

	Returns:
		tuple: The new state of the robot, as a tuple of its x and y coordinates and its orientation.

	"""
	x_pos, y_pos, theta = state
	#check action is valid
	action_to_do = action_operator.get(type_action, None)
	if action_to_do is None:
		return None
	#get the proper orientation of the robot, its current front
	value_angle = angle_values_real[action_to_do]
	rotation_index = int(theta * factor_angle)
	vector_front = np.dot(rotation_angle_matrices[rotation_index], normal_x_vector)
	new_vector = np.dot(rotation_angle_matrices[action_to_do], vector_front) * step_size
	#calculate new positions and orientation
	x_pos_new = round_float(round(x_pos + new_vector[0], 2))
	y_pos_new = round_float(round(y_pos + new_vector[1], 2))
	angle_degrees = theta + value_angle
	if angle_degrees > 180:
		angle_degrees = angle_degrees - 360
	if angle_degrees < -180:
		angle_degrees = angle_degrees + 360
	return (x_pos_new, y_pos_new, angle_degrees)

def convert_check_matrix(node):
	"""
	This function converts a node to its true position in the check matrix.

	Args:
		node (tuple): The node to convert, as a tuple of its x and y coordinates and its orientation.

	Returns:
		tuple: The node's true position in the check matrix, as a tuple of its x and y coordinates and its angle index.

	"""
	x_pos, y_pos, theta = node
	x_pos_true = int(x_pos * factor_distance)
	y_pos_true = int(y_pos * factor_distance)
	angle_index = int(theta * factor_angle)
	return (x_pos_true, y_pos_true, angle_index)

def add_to_check_matrix(node):
	"""
	This function adds a node to the check matrix.

	Args:
		node (tuple): The node to add, as a tuple of its x and y coordinates and its orientation.
	"""
	x_pos_true, y_pos_true, angle_index = convert_check_matrix(node)
	check_duplicates_space[x_pos_true, y_pos_true, angle_index] = 1
	#? testing for faster node unqueue
	# check_duplicates_space[x_pos_true, y_pos_true, angle_index + 1] = 1
    # check_duplicates_space[x_pos_true, y_pos_true, angle_index - 1] = 1
    # check_duplicates_space[x_pos_true, y_pos_true, :] = 1

def is_duplicate(node):
	"""
	This function checks if a node is already in the check matrix.

	Args:
		node (tuple): The node to check, as a tuple of its x and y coordinates and its orientation.
	"""
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
	#create new node
	new_cost_to_come = current_node[1] + step_size
	new_cost_to_go = distance(state_moved[0:2], goal_state[0:2]) #heuristic function
	new_total_cost =  new_cost_to_come + new_cost_to_go
	new_node = (new_total_cost, new_cost_to_come, new_cost_to_go) + (-1, current_node[3]) + state_moved
	return new_node

def check_goal_reached(node_a, goal):
	"""
	This function checks if the goal state has been reached.

	Args:
		node_a (tuple): The node to check, as a tuple of its x and y coordinates and its orientation.
	"""
	dist_centers = distance(node_a[0:2], goal[0:2])
	orientation_valid = np.abs(node_a[2] - goal[2]) <= 2*th_angle #30 degrees both directions
	center_robot_in_radius_goal = dist_centers < radius_goal**2
	if center_robot_in_radius_goal and orientation_valid:
		return True
	return False
#A* ALGORITHM FUNCTIONS
def create_nodes(initial_state, goal_state):
	"""Creates the State space of all possible movements until goal state is reached by applying the A* algorithm.

	Args:
			initial_state (array): multi dimensional array 3x3 that describes the initial configuarion of the puzzle
			goal_state (array): multi dimensional array 3x3 that describes the final configuration the algorithm must find.

	Returns:
			str: 'DONE'. The process have ended thus we have a solution in the tree structure generated.
	"""
	# Start the timer
	start_time = time.time()
	goal_reached = False
	counter_nodes = 0
	distance_init_goal = distance(initial_state[0:2], goal_state[0:2])
	cost_init = (distance_init_goal, 0, distance_init_goal)
	initial_node = cost_init + (0, None) + initial_state
	# Add initial node to the heap
	hq.heappush(generated_nodes, initial_node)
	while not goal_reached and len(generated_nodes) and not len(generated_nodes) > 150000:
		print(counter_nodes)
		current_node = generated_nodes[0]
		hq.heappop(generated_nodes)
		# Mark node as visited
		visited_nodes.append(current_node)
		#for check duplicates
		add_to_check_matrix(current_node[5:])
		visited_vectors[current_node[5:7]] = []
		# Check if popup_node is goal state
		goal_reached = check_goal_reached(current_node[5:], goal_state)
		if goal_reached:
			goal_reached = True
			end_time = time.time()
			return f'DONE in {end_time-start_time} seconds.'
		# Apply action set to node to get new states/children
		for action in action_set:
			child = action_move(current_node, action)
			# If movement was not possible, ignore it
			if not child:
				continue
			visited_vectors[current_node[5:7]].append(child[5:7])
			# Check if child is in open list generated nodes
			where_is_node = 0
			is_in_open_list = False
			for node in generated_nodes:
				if node[5:] == child[5:]:
					is_in_open_list = True
					break
				where_is_node += 1
			if not is_in_open_list:
				counter_nodes += 1
				child_to_enter = child[0:3] + (counter_nodes,) + child[4:]
				hq.heappush(generated_nodes, child_to_enter)
			# check if cost to come is greater in node in open list
			elif generated_nodes[where_is_node][1] > child[1]:
				# Update parent node and cost of this node in the generated nodes heap
				current_index = generated_nodes[where_is_node][3]
				generated_nodes[where_is_node] = child[0:3] + (current_index,) + child[4:]
	end_time = time.time()
	print(f'No solution found. Process took {end_time-start_time} seconds.')
	return None

def generate_path(node):
	"""Generate the path from the initial node to the goal state.

	Args:
		node (Node): Current node to evaluate its parent (previous move done).
	Returns:
		Boolean: True if no more of the path are available
	"""
	while node is not None:
		goal_path.append(node[5:])
		parent_at = 0
		for node_check in visited_nodes:
			if node_check[3] == node[4]:
				break
			parent_at += 1
		node = visited_nodes[parent_at] if parent_at < len(visited_nodes) else None
	return True


#USER VARIABLES
#*define input and goal coordinates
initial_state = coordinate_input('initial')
goal_state = coordinate_input('goal')
#*define robot parameters
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
visited_nodes = []  # full list node visited
visited_vectors = {} # for animation
goal_path = [] #nodes shortest path
hq.heapify(generated_nodes)

#*matrxi to check for duplicate nodes
check_duplicates_space = np.zeros(
	(rows_check_space, cols_check_space, angle_check_space))
#* Provides matrices of rotation to instead apply sin/cos
rotation_angle_matrices = np.array([ rotation_vectors_by(angle) for angle in angle_values_real ])

solution = create_nodes(initial_state, goal_state)
print(solution)
if not solution:
	exit(0)

generate_path(visited_nodes[-1])#modifies goal path
#for animation of backtracking shortest path
goal_path_animation = np.array(goal_path)
x_shortest = goal_path_animation[:,0]
y_shortest = goal_path_animation[:,1]

#ANIMATION
