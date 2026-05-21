import math
import heapq

#from this node, what is straight-line distnace to the nearest goal
def heuristic(node, nodes, destinations): # current node number, dictionary of node coordinates, list of goal nodes
    x1, y1 = nodes[node] 
    best_distance = float("inf") 

    #finds closest destination 
    for goal in destinations:
        x2, y2 = nodes[goal]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if distance < best_distance:
            best_distance = distance #distnace to nearest destination

    return best_distance

def a_star_search(graph, nodes, origin, destinations):
    frontier = []
    visited = set() #unordered set of notes visited, dont want to expand the same node again
    parent = {origin: None} # where each node came from

    # stores g(n), which is the actual path cost from start to that node
    g_cost = {origin: 0}

    number_of_nodes = 1 #node count
    insertion_order = 0 # tie-breaker

    # calculates h(n) for the starting node
    start_h = heuristic(origin, nodes, destinations)

    start_f = start_h

    # pushes the origin into the heap
    # tuple format:
    # (f_cost, node_number, insertion_order, current_node, path)
    heapq.heappush(frontier, (start_f, origin, insertion_order, origin, [origin]))

    # continues searching while there are still nodes in the frontier
    while frontier:
        # pops the best node based on smallest f(n)
        f, node_number, order, current_node, path = heapq.heappop(frontier)

        # if already expanded before, skip it
        if current_node in visited:
            continue

        # marks this node as expanded
        visited.add(current_node)

        # if current node is one of the goals, return the result
        if current_node in destinations:
            return current_node, number_of_nodes, path

        # tie-breaking
        neighbours = sorted(graph[current_node], key=lambda x: x[0])

        for neighbour, cost in neighbours:
            if neighbour in visited:
                continue

            # calculates new g(n) = cost so far + edge cost
            new_g = g_cost[current_node] + cost

            # only update if this is the first time seeing the neighbour
            # or if this path is cheaper than the old one
            if neighbour not in g_cost or new_g < g_cost[neighbour]:
                g_cost[neighbour] = new_g

                # remember how we reached this neighbour
                parent[neighbour] = current_node
                insertion_order += 1

                # calculate h(n) for this neighbour
                neighbour_h = heuristic(neighbour, nodes, destinations)

                # calculate f(n) = g(n) + h(n)
                neighbour_f = new_g + neighbour_h

                # build the path to this neighbour
                new_path = path + [neighbour]

                # add neighbour to the frontier
                heapq.heappush(
                    frontier,
                    (neighbour_f, neighbour, insertion_order, neighbour, new_path)
                )
 
                number_of_nodes += 1

    # if no goal can be reached, return failure
    return None, number_of_nodes, []