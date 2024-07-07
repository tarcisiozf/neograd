package main

import (
	"fmt"
	"neograd/engine"
	"neograd/neuralnet"
)

func main() {
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

	mlp := neuralnet.NewMultiLayerPerceptron(len(inputs[0]), 4, 4, 1).Activation(engine.Tanh)

	predictions := make([]*engine.Value, len(inputs))

	// gradient descent
	for i := 0; i < 300; i++ {
		// forward pass
		for i, input := range inputs {
			predictions[i] = mlp.Call(input).Item()
		}

		loss := neuralnet.MeanSquaredError(desiredTargets, predictions).Label("loss")

		// backward pass
		for _, param := range mlp.Parameters() {
			param.ZeroGrad()
		}
		loss.Backward()

		// update
		for _, param := range mlp.Parameters() {
			param.Adjust(param.Grad() * -0.01)
		}

		fmt.Println(loss)
	}

	fmt.Println(predictions)
}
