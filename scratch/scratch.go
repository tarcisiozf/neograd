package main

import (
	"math"
	"neograd/engine"
)

var w1, b1, w2, b2 [][]float32

func randn(rows, cols int) [][]float32 {
	mat := make([][]float32, rows)
	for i := range mat {
		mat[i] = make([]float32, cols)
		for j := range mat[i] {
			mat[i][j] = engine.RandomUniform(-1, 1)
		}
	}
	return mat
}

func dot(a, b [][]float32) [][]float32 {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if colsA != rowsB {
		panic("matrix dimensions do not match")
	}

	c := make([][]float32, rowsA)
	for i := range c {
		c[i] = make([]float32, colsB)
		for j := range c[i] {
			for k := 0; k < colsA; k++ {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return c
}

func add(a, b [][]float32) [][]float32 {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if rowsA != rowsB || colsA != colsB {
		panic("matrix dimensions do not match")
	}

	c := make([][]float32, rowsA)
	for i := range c {
		c[i] = make([]float32, colsA)
		for j := range c[i] {
			c[i][j] = a[i][j] + b[i][j]
		}
	}
	return c
}

func ReLU(a [][]float32) [][]float32 {
	rows, cols := len(a), len(a[0])
	c := make([][]float32, rows)
	for i := range c {
		c[i] = make([]float32, cols)
		for j := range c[i] {
			c[i][j] = max(a[i][j], 0)
		}
	}
	return c
}

func softmax(a [][]float32) [][]float32 {
	rows, cols := len(a), len(a[0])
	c := make([][]float32, rows)
	for i := range c {
		c[i] = make([]float32, cols)
		for j := range c[i] {
			c[i][j] = float32(math.Exp(float64(a[i][j]))) // TODO: wrong
		}
	}
	return c
}

func forwardProp(X [][]float32) ([][]float32, [][]float32, [][]float32, [][]float32) {
	z1 := add(dot(w1, X), b1)
	a1 := ReLU(z1)
	z2 := add(dot(w2, a1), b2)
	a2 := softmax(z2)
	return z1, a1, z2, a2
}

func lmax(a ...float32) float32 {
	m := a[0]
	for _, v := range a {
		if v > m {
			m = v
		}
	}
	return m
}

func oneHot(y [][]float32) [][]float32 {
	rows, cols := len(y[0]), int(lmax(y[0]...))+1
	c := make([][]float32, rows)
	for i := range c {
		c[i] = make([]float32, cols)
		c[i][int(y[0][i])] = 1
	}
	return transpose(c)
}

func transpose(c [][]float32) [][]float32 {
	rows, cols := len(c), len(c[0])
	t := make([][]float32, cols)
	for i := range t {
		t[i] = make([]float32, rows)
		for j := range t[i] {
			t[i][j] = c[j][i]
		}
	}
	return t
}

func main() {
	// init params:
	w1 = randn(10, 784)
	b1 = randn(10, 1)
	w2 = randn(10, 10)
	b2 = randn(10, 1)

	y := [][]float32{{1, 2, 3}}
	oh := oneHot(y)

	println(oh)
}
