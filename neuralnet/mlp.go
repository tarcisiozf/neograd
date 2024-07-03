package neuralnet

import (
	"neograd/engine"
)

type MultiLayerPerceptron struct {
	layers []*Layer
}

func NewMultiLayerPerceptron(inputs ...int) *MultiLayerPerceptron {
	layers := make([]*Layer, len(inputs)-1)
	for i := 0; i < len(inputs)-1; i++ {
		layers[i] = NewLayer(inputs[i], inputs[i+1])
	}

	return &MultiLayerPerceptron{
		layers: layers,
	}
}

func (m *MultiLayerPerceptron) Call(input []*engine.Value) engine.ValueList {
	for _, layer := range m.layers {
		input = layer.Call(input)
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
