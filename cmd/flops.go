package main

import "fmt"

func calcFlops(inputs ...int) {
	var mul, add, bias, act int

	for i := 0; i < len(inputs)-1; i++ {
		numInputs := inputs[i]
		numOutputs := inputs[i+1]

		// calculate operations for matrix multiplication
		mul += numOutputs * numInputs
		add += numOutputs * (numInputs - 1)

		bias += numOutputs // bias
		act += numOutputs  // f()
	}

	fmt.Println(
		"mul: ", mul,
		"add: ", add,
		"bias: ", bias,
		"act: ", act,
		"total: ", mul+add+bias+act,
	)
}

func main() {
	calcFlops(2, 3, 2, 1)
	calcFlops(28*28, 800, 10)
}
