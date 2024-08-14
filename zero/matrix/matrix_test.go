package matrix_test

import (
	"neograd/zero/matrix"
	"testing"
)

func TestMatrix_Add(t *testing.T) {
	a := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	b := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	c := a.Add(b).Internal()
	if c[0][0] != 2 || c[0][1] != 4 || c[1][0] != 6 || c[1][1] != 8 {
		t.Errorf("Matrix addition failed")
	}
}

func TestMatrix_Dot(t *testing.T) {
	a := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	b := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	c := a.Dot(b).Internal()
	if c[0][0] != 7 || c[0][1] != 10 || c[1][0] != 15 || c[1][1] != 22 {
		t.Errorf("Matrix dot product failed")
	}
}

func TestRandom(t *testing.T) {
	n := 2
	out := matrix.Random(n, n).Internal()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if out[i][j] < -0.5 || out[i][j] > 0.5 {
				t.Errorf("Random matrix failed")
			}
		}
	}
}

func TestReLU(t *testing.T) {
	a := matrix.FromSlice([][]float32{
		{1, -2},
		{3, -4},
	})
	b := matrix.ReLU(a).Internal()
	if b[0][0] != 1 || b[0][1] != 0 || b[1][0] != 3 || b[1][1] != 0 {
		t.Errorf("ReLU failed")
	}
}

func TestSoftmax(t *testing.T) {
	a := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	b := matrix.Softmax(a).Internal()
	if b[0][0] != 0.032058604 || b[0][1] != 0.08714432 || b[1][0] != 0.23688282 || b[1][1] != 0.6439142 {
		t.Errorf("Softmax failed")
	}
}

func TestMatrix_Transpose(t *testing.T) {
	a := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	b := a.Transpose().Internal()
	if b[0][0] != 1 || b[0][1] != 3 || b[1][0] != 2 || b[1][1] != 4 {
		t.Errorf("Matrix transpose failed")
	}
}

func TestOneHot(t *testing.T) {
	a := []float32{1, 0, 3}
	b := matrix.OneHot(a).Internal()
	if b[0][0] != 0 || b[0][1] != 1 || b[0][2] != 0 {
		t.Errorf("One-hot failed")
	}
	if b[1][0] != 1 || b[1][1] != 0 || b[1][2] != 0 {
		t.Errorf("One-hot failed")
	}
	if b[2][0] != 0 || b[2][1] != 0 || b[2][2] != 0 {
		t.Errorf("One-hot failed")
	}
	if b[3][0] != 0 || b[3][1] != 0 || b[3][2] != 1 {
		t.Errorf("One-hot failed")
	}
}

func TestMatrix_Sub(t *testing.T) {
	a := matrix.FromSlice([][]float32{
		{1, 2},
		{3, 4},
	})
	c := a.Sub(a).Internal()
	if c[0][0] != 0 || c[0][1] != 0 || c[1][0] != 0 || c[1][1] != 0 {
		t.Errorf("Matrix subtraction failed")
	}
}
