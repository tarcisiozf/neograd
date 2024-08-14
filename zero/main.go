package main

import (
	"neograd/zero/matrix"
)

func main() {
	gradientDescent(matrix.Random(784, 3), []float32{1, 2, 9}, 1, 0.01)
}

func initParams() (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	w1 := matrix.Random(10, 784)
	b1 := matrix.Random(10, 1)
	w2 := matrix.Random(10, 10)
	b2 := matrix.Random(10, 1)
	return w1, b1, w2, b2
}

func forwardPass(w1, b1, w2, b2, X *matrix.Matrix) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	z1 := w1.Dot(X).Add(b1)
	a1 := matrix.ReLU(z1)
	z2 := w2.Dot(a1).Add(b2)
	a2 := matrix.Softmax(z2)
	return z1, a1, z2, a2
}

func derivReLU(z *matrix.Matrix) *matrix.Matrix {
	out := matrix.FromShape(z)
	v := out.Internal()
	for i := range v {
		for j := range v[i] {
			if v[i][j] > 0 {
				out.Set(i, j, 1)
			}
		}
	}
	return out
}

func backProp(z1, a1, z2, a2, w2, x *matrix.Matrix, y []float32) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	m := len(y)
	ohY := matrix.OneHot(y)
	dz2 := a2.Sub(ohY)
	dw2 := matrix.Mulf(dz2.Dot(a1.Transpose()), 1/float32(m))
	db2 := matrix.Mulf(dz2.Sum(2), 1/float32(m)) // TODO: dim of video
	dz1 := w2.Transpose().Dot(dz2).Mul(derivReLU(z1))
	dw1 := matrix.Mulf(dz1.Dot(x.Transpose()), 1/float32(m))
	db1 := matrix.Mulf(dz1.Sum(2), 1/float32(m)) // TODO: dim of video
	return dw1, db1, dw2, db2
}

func updateParams(w1, b1, w2, b2, dw1, db1, dw2, db2 *matrix.Matrix, lr float32) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	w1 = w1.Sub(matrix.Mulf(dw1, lr))
	b1 = b1.Sub(matrix.Mulf(db1, lr))
	w2 = w2.Sub(matrix.Mulf(dw2, lr))
	b2 = b2.Sub(matrix.Mulf(db2, lr))
	return w1, b1, w2, b2
}

func gradientDescent(x *matrix.Matrix, y []float32, iterations int, lr float32) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	w1, b1, w2, b2 := initParams()
	for i := 0; i < iterations; i++ {
		z1, a1, z2, a2 := forwardPass(w1, b1, w2, b2, x)
		dw1, db1, dw2, db2 := backProp(z1, a1, z2, a2, w2, x, y)
		w1, b1, w2, b2 = updateParams(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)
	}
	return w1, b1, w2, b2
}
