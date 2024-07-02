package engine

func TopologicalSort(start *Value) []*Value {
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)
	queue := []*Value{start}
	var current *Value

	for len(queue) > 0 {
		current, queue = queue[0], queue[1:]
		if visited[current] {
			continue
		}
		visited[current] = true
		for _, child := range current.prev {
			if !visited[child] {
				queue = append(queue, child)
			}
		}
		topo = append(topo, current)
	}

	return topo
}
