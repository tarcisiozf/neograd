package neuralnet_test

import (
	"neograd/engine"
	"neograd/neuralnet"
	"testing"
)

func TestMultiLayerPerceptron_Call(t *testing.T) {
	input := []*engine.Value{
		engine.NewValue(2),
		engine.NewValue(3),
		engine.NewValue(-1),
	}
	numInputs := len(input)
	mlp := neuralnet.NewMultiLayerPerceptron(numInputs, 4, 4, 1)
	result := mlp.Call(input)

	if len(result) != 1 {
		t.Error("Expected 1 result, got ", len(result))
	}
	if result[0] == nil {
		t.Error("Expected non-nil result, got ", result[0])
	}
}
