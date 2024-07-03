package neuralnet_test

import (
	"neograd/engine"
	"neograd/neuralnet"
	"testing"
)

func TestMeanSquaredError(t *testing.T) {
	inputs := [][]*engine.Value{
		{engine.NewValue(2), engine.NewValue(3), engine.NewValue(-1)},
		{engine.NewValue(3), engine.NewValue(-1), engine.NewValue(0.5)},
		{engine.NewValue(0.5), engine.NewValue(1), engine.NewValue(1)},
		{engine.NewValue(1), engine.NewValue(1), engine.NewValue(-1)},
	}
	desiredTargets := []*engine.Value{
		engine.NewValue(1),
		engine.NewValue(-1),
		engine.NewValue(-1),
		engine.NewValue(1),
	}

	mlp := neuralnet.NewMultiLayerPerceptron(len(inputs[0]), 4, 4, 1)

	predictions := make([]*engine.Value, len(inputs))
	for i, input := range inputs {
		predictions[i] = mlp.Call(input).Item()
	}

	loss := neuralnet.MeanSquaredError(desiredTargets, predictions)
	if loss == nil {
		t.Fatal("Expected error, got nil")
	}
	loss.Backward()
}
