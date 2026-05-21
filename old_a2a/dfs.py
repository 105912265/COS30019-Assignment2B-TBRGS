def depth_first_search(graph, origin, destinations):
    visited = set()
    stack = [(origin, [origin], 0)]  # (current, path, cost)
    nodes_created = 1

    visited.add(origin)

    while stack:
        current, path, cost = stack.pop()

        if current in destinations:
            return current, nodes_created, path

        for n, edge_cost in reversed(sorted(graph[current])):  # tie-breaking
            if n not in visited:
                visited.add(n)
                stack.append((n, path + [n], cost + edge_cost))
                nodes_created += 1

    return None, nodes_created, []