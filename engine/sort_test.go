package engine

import "testing"

func TestTopologicalSort(t *testing.T) {
	a := NewValue(1).Label("a")
	b := NewValue(2).Label("b")
	c := a.Add(b).Label("c")
	d := NewValue(3).Label("d")
	e := c.Mul(d).Label("e")

	result := TopologicalSort(e)
	if len(result) != 5 {
		t.Errorf("Topological sort returned wrong length: got %d, want %d", len(result), 5)
	}
	if result[0] != e {
		t.Errorf("Topological sort returned wrong value: got %v, want %v", result[0], e)
	}
	if result[1] != c {
		t.Errorf("Topological sort returned wrong value: got %v, want %v", result[1], c)
	}
	if result[2] != d {
		t.Errorf("Topological sort returned wrong value: got %v, want %v", result[2], d)
	}
	if result[3] != a {
		t.Errorf("Topological sort returned wrong value: got %v, want %v", result[3], a)
	}
	if result[4] != b {
		t.Errorf("Topological sort returned wrong value: got %v, want %v", result[4], b)
	}
}
