import heapq


def astar_search(graph, start, goal):
    """
    Search for the lowest-travel-time path from start to goal.

    graph format:
        {
            2000: [(2200, 2.5), (2820, 3.1)],
            2200: [(3002, 4.7)],
            3002: []
        }

    The second value in each edge tuple is travel time in minutes.
    """

    frontier = []
    heapq.heappush(frontier, (0, start, [start], 0))

    visited = set()

    while frontier:
        _, current, path, cost_so_far = heapq.heappop(frontier)

        if current == goal:
            return path, cost_so_far

        if current in visited:
            continue

        visited.add(current)

        for neighbour, travel_time in graph.get(current, []):
            if neighbour in visited:
                continue

            new_cost = cost_so_far + travel_time

            # Heuristic is set to 0 for the dummy integration graph.
            # This behaves like uniform-cost search while keeping the A* structure.
            heuristic = 0

            heapq.heappush(
                frontier,
                (new_cost + heuristic, neighbour, path + [neighbour], new_cost),
            )

    return None, float("inf")
def find_top_k_paths(graph, start, goal, k=5):
    routes = []
    frontier = [(0, start, [start])]

    while frontier and len(routes) < k:
        cost_so_far, current, path = heapq.heappop(frontier)

        if current == goal:
            routes.append((path, cost_so_far))
            continue

        for neighbour, travel_time in graph.get(current, []):
            if neighbour not in path:
                new_path = path + [neighbour]
                new_cost = cost_so_far + travel_time
                heapq.heappush(frontier, (new_cost, neighbour, new_path))

    return routes

if __name__ == "__main__":
    test_graph = {
        2000: [(2200, 3.0), (2820, 4.5)],
        2200: [(3002, 5.0)],
        2820: [(3002, 2.0)],
        3002: [],
    }

    path, cost = astar_search(test_graph, 2000, 3002)

    print("Best path:", path)
    print("Travel time:", round(cost, 2), "minutes")

    routes = find_top_k_paths(test_graph, 2000, 3002, k=5)

    print("\nTop routes:")
    for index, (route, total_time) in enumerate(routes, start=1):
        print(f"{index}. Path: {route}")
        print(f"   Travel time: {round(total_time, 2)} minutes")