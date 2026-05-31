import heapq


def astar_search(graph, origin, destination, banned_edges=None):

    #banned_edges is used when finding alternative routes.


    if banned_edges is None:
        banned_edges = set()

    origin = int(origin)
    destination = int(destination)

    priority_queue = [(0, origin, [origin])]
    visited = set()

    while priority_queue:
        total_cost, current_node, path = heapq.heappop(priority_queue)

        if current_node == destination:
            return path, total_cost

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbour, edge_cost in graph.get(current_node, []):
            edge = (current_node, neighbour)

            if edge in banned_edges:
                continue

            if neighbour not in visited:
                new_cost = total_cost + edge_cost
                new_path = path + [neighbour]

                heapq.heappush(
                    priority_queue,
                    (new_cost, neighbour, new_path)
                )

    return None, float("inf")


def find_top_k_routes(graph, origin, destination, k=5):
    """
    Finds up to k routes using repeated A*.

    Route 1 is the normal A* route.
    For each next route, one edge from a previous route is temporarily banned,
    then A* is run again to find an alternative path.
    """

    routes = []
    banned_edges_sets = [set()]
    tried_paths = set()

    while banned_edges_sets and len(routes) < k:
        banned_edges = banned_edges_sets.pop(0)

        path, total_time = astar_search(
            graph=graph,
            origin=origin,
            destination=destination,
            banned_edges=banned_edges
        )

        if path is None:
            continue

        path_tuple = tuple(path)

        if path_tuple in tried_paths:
            continue

        tried_paths.add(path_tuple)
        routes.append((path, total_time))

        # Create alternative searches by banning one edge from this route
        for i in range(len(path) - 1):
            new_banned_edges = set(banned_edges)
            new_banned_edges.add((path[i], path[i + 1]))
            banned_edges_sets.append(new_banned_edges)

    routes.sort(key=lambda route: route[1])

    return routes[:k]