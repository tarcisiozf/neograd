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
	if b[0][0] != 0.0320586 || b[0][1] != 0.08714432 || b[1][0] != 0.23688282 || b[1][1] != 0.64391426 {
		t.Errorf("Softmax failed")
	}
}
