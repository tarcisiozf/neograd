package mnist

import (
	"neograd/engine"
	"neograd/neuralnet"
	"testing"
)

const inputSize = 28 * 28

func createInputs(size int) []*engine.Value {
	out := make([]*engine.Value, size)
	for i := 0; i < size; i++ {
		out[i] = engine.NewValue(0)
	}
	return out
}

func BenchmarkGraph(b *testing.B) {
	input := createInputs(inputSize)
	mlp := neuralnet.NewMultiLayerPerceptron(inputSize, 800, 10, 1)

	for i := 0; i < b.N; i++ {
		mlp.Call(input)
	}
}

func BenchmarkMatmul(b *testing.B) {
	input := createInputs(inputSize)
	mlp := neuralnet.NewMultiLayerPerceptron(inputSize, 800, 10, 1)

	for i := 0; i < b.N; i++ {
		mlp.CallMatrix(input)
	}
}

func BenchmarkParallel(b *testing.B) {
	input := createInputs(inputSize)
	mlp := neuralnet.NewMultiLayerPerceptron(inputSize, 800, 10, 1)

	for i := 0; i < b.N; i++ {
		mlp.CallParallel(input)
	}
}
