package main

import (
	"math"
	"neograd/engine"
)

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
		//panic("matrix dimensions do not match")
		b = broadcast(b, rowsA, colsA)
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

func broadcast(a [][]float32, rows int, cols int) [][]float32 {
	if rows != len(a) {
		panic("rows do not match")
	}
	c := make([][]float32, rows)
	for i := range c {
		c[i] = make([]float32, cols)
		for j := range c[i] {
			c[i][j] = a[i][0]
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

func forwardProp(w1, b1, w2, b2, X [][]float32) ([][]float32, [][]float32, [][]float32, [][]float32) {
	z1 := add(dot(w1, X), b1)
	a1 := ReLU(z1)
	z2 := add(dot(w2, a1), b2)
	a2 := softmax(z2)
	return z1, a1, z2, a2
}

func lmax(a [][]float32) float32 {
	m := a[0][0]
	for _, v := range a {
		if v[0] > m {
			m = v[0]
		}
	}
	return m
}

func oneHot(y [][]float32) [][]float32 {
	rows, cols := size(y), int(lmax(y))+1
	c := make([][]float32, rows)
	for i := range c {
		c[i] = make([]float32, cols)
		c[i][int(y[i][0])] = 1
	}
	return transpose(c)
}

func size(y [][]float32) int {
	return len(y) * len(y[0])
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

func derivReLU(z [][]float32) [][]float32 {
	c := make([][]float32, len(z))
	for i := range z {
		c[i] = make([]float32, len(z[i]))
		for j := range z[i] {
			if z[i][j] > 0 {
				c[i][j] = 1
			} else {
				c[i][j] = 0
			}
		}
	}
	return c
}

func backProp(z1, a1, z2, a2, w2, X, Y [][]float32) ([][]float32, [][]float32, [][]float32, [][]float32) {
	m := size(Y)
	ohY := oneHot(Y)
	dZ2 := sub(a2, ohY)
	dW2 := mulf(1/float32(m), dot(dZ2, transpose(a1)))
	db2 := mulf(1/float32(m), sum(dZ2, 2))
	dZ1 := mul(derivReLU(z1), dot(transpose(w2), dZ2))
	dW1 := mulf(1/float32(m), dot(dZ1, transpose(X)))
	db1 := mulf(1/float32(m), sum(dZ1, 2))
	return dW1, db1, dW2, db2
}

func mul(a [][]float32, b [][]float32) [][]float32 {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if rowsA != rowsB || colsA != colsB {
		panic("matrix dimensions do not match")
	}

	c := make([][]float32, rowsA)
	for i := range c {
		c[i] = make([]float32, colsA)
		for j := range c[i] {
			c[i][j] = a[i][j] * b[i][j]
		}
	}
	return c
}

func mulf(f float32, m [][]float32) [][]float32 {
	rows, cols := len(m), len(m[0])
	c := make([][]float32, rows)
	for i := range c {
		c[i] = make([]float32, cols)
		for j := range c[i] {
			c[i][j] = f * m[i][j]
		}
	}
	return c
}

func sum(a [][]float32, dim int) [][]float32 {
	rows, cols := len(a), len(a[0])
	var c [][]float32
	if dim == 1 {
		c = make([][]float32, rows)
		for i := range c {
			c[i] = make([]float32, 1)
			for j := range c[i] {
				for k := range a[i] {
					c[i][j] += a[i][k]
				}
			}
		}
	} else if dim == 2 {
		c = make([][]float32, 1)
		c[0] = make([]float32, cols)
		for i := range c[0] {
			for j := range a {
				c[0][i] += a[j][i]
			}
		}
	}
	return c
}

func sub(a [][]float32, b [][]float32) [][]float32 {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if rowsA != rowsB || colsA != colsB {
		panic("matrix dimensions do not match")
	}

	c := make([][]float32, rowsA)
	for i := range c {
		c[i] = make([]float32, colsA)
		for j := range c[i] {
			c[i][j] = a[i][j] - b[i][j]
		}
	}
	return c
}

func updateParams(w1, b1, w2, b2, dw1, db1, dw2, db2 [][]float32, learningRate float32) ([][]float32, [][]float32, [][]float32, [][]float32) {
	w1 = sub(w1, mulf(learningRate, dw1))
	b1 = sub(b1, transpose(mulf(learningRate, db1)))
	w2 = sub(w2, mulf(learningRate, dw2))
	b2 = sub(b2, transpose(mulf(learningRate, db2)))
	return w1, b1, w2, b2
}

func gradientDescent(X, Y [][]float32, iterations int, learningRate float32) {
	// init params:
	w1 := randn(10, 784)
	b1 := randn(10, 1)
	w2 := randn(10, 10)
	b2 := randn(10, 1)

	for i := 0; i < iterations; i++ {
		z1, a1, z2, a2 := forwardProp(w1, b1, w2, b2, X)
		dw1, db1, dw2, db2 := backProp(z1, a1, z2, a2, w2, X, Y)
		w1, b1, w2, b2 = updateParams(w1, b1, w2, b2, dw1, db1, dw2, db2, learningRate)
	}
}

func main() {
	data := make([][]float32, 10)
	for i := range data {
		data[i] = make([]float32, 784)
		for j := range data[i] {
			data[i][j] = engine.RandomUniform(-1, 1)
		}
	}

	X := transpose(data)
	Y := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		Y[i] = []float32{float32(i)}
	}

	gradientDescent(X, Y, 100, 0.1)
}
