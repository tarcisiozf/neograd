package engine

import (
	"fmt"
	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"log"
)

func trace(root *Value) ([]*Value, [][2]int) {
	visited := make(map[*Value]bool)
	nodes := make([]*Value, 0)
	edges := make([][2]int, 0)

	id := func(v *Value) int {
		for i, node := range nodes {
			if node == v {
				return i
			}
		}
		nodes = append(nodes, v)
		return len(nodes) - 1
	}

	var build func(v *Value)
	build = func(v *Value) {
		if _, ok := visited[v]; !ok {
			visited[v] = true
			for _, child := range v.prev {
				edges = append(edges, [2]int{id(child), id(v)})
				build(child)
			}
		}
	}
	build(root)

	return nodes, edges
}

func Draw(root *Value) error {
	g := graphviz.New()
	graph, err := g.Graph(graphviz.Directed)
	if err != nil {
		return err
	}
	defer func() {
		if err := graph.Close(); err != nil {
			log.Fatal(err)
		}
		if err := g.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	values, edges := trace(root)

	nodes := make([]*cgraph.Node, len(values))
	for i, v := range values {
		label := fmt.Sprintf("%s data=%.4f grad=%.4f", v.label, v.data, v.grad)
		nodes[i], err = graph.CreateNode(label)
		if err != nil {
			return err
		}
	}

	for _, edge := range edges {
		src, dst := nodes[edge[0]], nodes[edge[1]]
		label := values[edge[1]].label
		e, err := graph.CreateEdge(label, src, dst)
		e.SetLabel(label)
		if err != nil {
			return err
		}
	}

	// 3. write to file directly
	if err := g.RenderFilename(graph, graphviz.PNG, "./graph.png"); err != nil {
		return err
	}

	return nil
}
