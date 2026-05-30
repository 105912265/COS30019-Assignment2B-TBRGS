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


if __name__ == "__main__":
    test_graph = {
        2000: [(2200, 3.0), (2820, 4.5)],
        2200: [(3002, 5.0)],
        2820: [(3002, 2.0)],
        3002: [],
    }

    path, cost = astar_search(test_graph, 2000, 3002)

    print("Path:", path)
    print("Travel time:", round(cost, 2), "minutes")
