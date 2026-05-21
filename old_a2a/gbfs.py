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


def greedy_best_first_search(graph, nodes, origin, destinations): #edges, dictionary of node coordinates,, starting node, goal nodes
    frontier = []
    visited = set() #unordered set of notes visited, dont want to expand the same node again
    parent = {origin: None} # where each node came from
    number_of_nodes = 1 #node count
    insertion_order = 0 # tie-breaker

    start_h = heuristic(origin, nodes, destinations)

    # (heuristic, node number, insertion order, current node, path)
    heapq.heappush(frontier, (start_h, origin, insertion_order, origin, [origin]))

    # keep searching while there are nodes in the frontier
    while frontier:
        h, node_number, order, current_node, path = heapq.heappop(frontier)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node in destinations:
            return current_node, number_of_nodes, path

        neighbours = sorted(graph[current_node], key=lambda x: x[0])

        for neighbour, cost in neighbours:
            if neighbour not in visited and neighbour not in parent:
                parent[neighbour] = current_node
                insertion_order += 1
                neighbour_h = heuristic(neighbour, nodes, destinations)
                new_path = path + [neighbour]

                heapq.heappush(
                    frontier,
                    (neighbour_h, neighbour, insertion_order, neighbour, new_path)
                )
                number_of_nodes += 1

    return None, number_of_nodes, []