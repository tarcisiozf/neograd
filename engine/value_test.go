package engine_test

import (
	"neograd/engine"
	"testing"
)

func TestValue_Add(t *testing.T) {
	a := engine.NewValue(2)
	b := engine.NewValue(3)
	result := a.Add(b)

	if result.Data() != 5 {
		t.Error("Expected 5, got ", result.Data())
	}
}

func TestValue_Mul(t *testing.T) {
	a := engine.NewValue(2)
	b := engine.NewValue(3)
	result := a.Mul(b)

	if result.Data() != 6 {
		t.Error("Expected 6, got ", result.Data())
	}
}

func TestValue_Backward(t *testing.T) {
	x1 := engine.NewValue(2)
	x2 := engine.NewValue(0)
	w1 := engine.NewValue(-3)
	w2 := engine.NewValue(1)
	b := engine.NewValue(6.8813735870195432)
	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)
	x1w1x2w2 := x1w1.Add(x2w2)
	n := x1w1x2w2.Add(b)
	o := n.Tanh()

	o.Backward()

	if o.Data() != 0.7071067 {
		t.Error("Expected 0.7071067, got ", o.Data())
	}
	if o.Grad() != 1 {
		t.Error("Expected 1, got ", o.Grad())
	}
	if x1.Grad() != -1.5000004 {
		t.Error("Expected -1.5000004, got ", x1.Grad())
	}
	if x2.Grad() != 0.5000001 {
		t.Error("Expected 0.5000001, got ", x2.Grad())
	}
}
