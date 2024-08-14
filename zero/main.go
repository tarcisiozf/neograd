package main

import (
	"neograd/zero/matrix"
)

func main() {

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

func backProp(z1, a1, z2, a2, w2 *matrix.Matrix, y []float32) {
	ohY := matrix.OneHot(y)
	dz2 := a2.Sub(ohY)
}
