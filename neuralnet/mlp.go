package neuralnet

import (
	"math"
	"neograd/engine"
	"sync"
)

func matrixMultiply(A, B []float32, rowsA, colsA, colsB int) []float32 {
	C := make([]float32, rowsA*colsB)

	// Perform matrix multiplication
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				C[i*colsB+j] += A[i*colsA+k] * B[k*colsB+j]
			}
		}
	}

	return C
}

type MultiLayerPerceptron struct {
	layers []*Layer
}

func NewMultiLayerPerceptron(inputs ...int) *MultiLayerPerceptron {
	layers := make([]*Layer, 0)
	for i := 0; i < len(inputs)-1; i++ {
		layers = append(layers, NewLayer(inputs[i], inputs[i+1]))
	}

	return &MultiLayerPerceptron{
		layers: layers,
	}
}

func (m *MultiLayerPerceptron) Add(layer *Layer) {
	m.layers = append(m.layers, layer)
}

func (m *MultiLayerPerceptron) Call(input []*engine.Value) engine.ValueList {
	for _, layer := range m.layers {
		input = layer.Call(input)
	}
	return input
}

func (m *MultiLayerPerceptron) CallMatrix(input engine.ValueList) engine.ValueList {
	for _, layer := range m.layers {
		weightsMatrix := make([]float32, len(layer.neurons)*len(input))
		for i, neuron := range layer.neurons {
			for j, weight := range neuron.weights {
				weightsMatrix[(i*len(input))+j] = weight.Data()
			}
		}
		result := matrixMultiply(weightsMatrix, input.DataSlice(), len(layer.neurons), len(input), 1)
		for i, neuron := range layer.neurons {
			result[i] = float32(math.Tanh(float64(result[i] + neuron.bias.Data())))
		}
		input = engine.ToList(result...)
	}
	return input
}

func (m *MultiLayerPerceptron) CallParallel(input engine.ValueList) engine.ValueList {
	wg := sync.WaitGroup{}

	for _, layer := range m.layers {
		result := make([]float32, len(layer.neurons))
		wg.Add(len(layer.neurons))

		for i, neuron := range layer.neurons {
			go func(i int, neuron *Neuron) {
				for j, weight := range neuron.weights {
					result[i] += weight.Data() * input[j].Data()
				}
				wg.Done()
			}(i, neuron)
		}
		wg.Wait()

		for i, neuron := range layer.neurons {
			result[i] = float32(math.Tanh(float64(result[i] + neuron.bias.Data())))
		}

		input = engine.ToList(result...)
	}

	return input
}

// TODO: use iterators
func (m *MultiLayerPerceptron) Parameters() []*engine.Value {
	params := make([]*engine.Value, 0)
	for _, layer := range m.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func (m *MultiLayerPerceptron) Activation(activator engine.ActivationFunction) *MultiLayerPerceptron {
	for _, layer := range m.layers {
		layer.Activation(activator)
	}
	return m
}
