package neuralnet

import "neograd/engine"

// TODO: add cross-entropy loss

func MeanSquaredError(groundTruth, predictions engine.ValueList) *engine.Value {
	sum := engine.NewValue(0)
	for i, gt := range groundTruth {
		sum = sum.Add(predictions[i].Sub(gt).Pow(2))
	}
	return sum
}
