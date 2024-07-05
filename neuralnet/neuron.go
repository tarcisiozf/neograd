package neuralnet

import (
	"neograd/engine"
)

type Neuron struct {
	bias    *engine.Value
	weights []*engine.Value
}

// TODO: replace with random uniform
func NewNeuron(numInputs int) *Neuron {
	weights := make([]*engine.Value, numInputs)
	for i := 0; i < numInputs; i++ {
		weights[i] = engine.NewValue(engine.RandomUniform(-1, 1))
	}
	bias := engine.NewValue(engine.RandomUniform(-1, 1))
	return &Neuron{
		bias:    bias,
		weights: weights,
	}
}

// f(w * x + b)
func (n *Neuron) Call(inputs []*engine.Value) *engine.Value {
	activation := n.bias
	for i, w := range n.weights {
		activation = activation.Add(inputs[i].Mul(w))
	}
	return activation.Tanh()
}

// TODO: use iterators
func (n *Neuron) Parameters() []*engine.Value {
	return append([]*engine.Value{n.bias}, n.weights...)
}
