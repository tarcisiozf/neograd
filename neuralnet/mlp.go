package neuralnet

import (
	"fmt"
	"neograd/engine"
)

type MultiLayerPerceptron struct {
	layers []*Layer
}

func NewMultiLayerPerceptron(inputs ...int) *MultiLayerPerceptron {
	layers := make([]*Layer, len(inputs)-1)
	for i := 0; i < len(inputs)-1; i++ {
		fmt.Println(inputs[i], inputs[i+1])
		layers[i] = NewLayer(inputs[i], inputs[i+1])
	}

	return &MultiLayerPerceptron{
		layers: layers,
	}
}

func (m *MultiLayerPerceptron) Call(input []*engine.Value) []*engine.Value {
	for _, layer := range m.layers {
		input = layer.Call(input)
	}
	return input
}
