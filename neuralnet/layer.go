package neuralnet

import "neograd/engine"

type Layer struct {
	neurons []*Neuron
}

func NewLayer(numInputs, numOutputs int) *Layer {
	neurons := make([]*Neuron, numOutputs)
	for i := range neurons {
		neurons[i] = NewNeuron(numInputs)
	}

	return &Layer{
		neurons: neurons,
	}
}

func (l *Layer) Call(inputs []*engine.Value) []*engine.Value {
	out := make([]*engine.Value, len(l.neurons))
	for i, neuron := range l.neurons {
		out[i] = neuron.Call(inputs)
	}
	return out
}
