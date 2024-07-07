package neuralnet

import "neograd/engine"

type Layer struct {
	neurons   []*Neuron
	activator engine.ActivationFunction
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

func (l *Layer) Activation(activator engine.ActivationFunction) *Layer {
	l.activator = activator
	return l
}

func (l *Layer) Call(inputs []*engine.Value) []*engine.Value {
	out := make([]*engine.Value, len(l.neurons))
	for i, neuron := range l.neurons {
		out[i] = l.activator(neuron.Call(inputs))
	}
	return out
}

// TODO: use iterators
func (l *Layer) Parameters() []*engine.Value {
	params := make([]*engine.Value, 0)
	for _, neuron := range l.neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}
