import heapq

def custom_search_1(graph, origin, destinations):
    visited = set()
    queue = [(0, origin, [origin])]  # (cost, node, path)
    nodes_created = 1

    while queue:
        cost, current, path = heapq.heappop(queue)

        if current in visited:
            continue
        visited.add(current)

        if current in destinations:
            return current, nodes_created, path

        for neighbour, edge_cost in sorted(graph[current]):
            if neighbour not in visited:
                heapq.heappush(queue, (cost + edge_cost, neighbour, path + [neighbour]))
                nodes_created += 1

    return None, nodes_created, []