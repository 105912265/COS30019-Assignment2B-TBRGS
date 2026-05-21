import heapq

def shortest_path_visit_all(graph, origin, destinations):
    # sort destinations so each one has a stable bit position
    destination_list = sorted(destinations)
    destination_index = {node: i for i, node in enumerate(destination_list)}

    # marks a destination as visited in the bitmask
    def update_mask(node, mask):
        if node in destination_index:
            bit = destination_index[node]
            mask |= (1 << bit)
        return mask

    # visiting none = 00
    # visiting 4 only = 01
    # visiting 5 only = 10
    # visiting both = 11
    goal_mask = (1 << len(destination_list)) - 1

    # starting mask depends on whether origin is already a destination
    start_mask = update_mask(origin, 0)

    # priority queue entries:
    # (total_cost, insertion_order, current_node, visited_mask, path)
    frontier = []
    insertion_order = 0
    heapq.heappush(frontier, (0, insertion_order, origin, start_mask, [origin]))

    # stores best known cost for each state
    # state = (current_node, visited_mask)
    best_cost = {(origin, start_mask): 0}

    # counts created states
    number_of_nodes = 1

    while frontier:
        cost_so_far, _, current_node, visited_mask, path = heapq.heappop(frontier)

        # skip old worse entries
        if best_cost.get((current_node, visited_mask), float("inf")) < cost_so_far:
            continue

        # if all destinations have been visited, return answer
        if visited_mask == goal_mask:
            return current_node, number_of_nodes, path

        # expand neighbours in ascending order
        neighbours = sorted(graph[current_node], key=lambda x: x[0])

        for neighbour, edge_cost in neighbours:
            new_cost = cost_so_far + edge_cost
            new_mask = update_mask(neighbour, visited_mask)
            new_state = (neighbour, new_mask)

            # only keep this state if it is better than before
            if new_state not in best_cost or new_cost < best_cost[new_state]:
                best_cost[new_state] = new_cost
                insertion_order += 1
                new_path = path + [neighbour]

                heapq.heappush(
                    frontier,
                    (new_cost, insertion_order, neighbour, new_mask, new_path)
                )
                number_of_nodes += 1

    return None, number_of_nodes, []