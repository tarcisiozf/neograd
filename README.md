# Neograd

A minimal deep learning framework written in Go. It includes:
•	A basic autograd engine
•	Feedforward neural network (MLP) implementation
•	Tokenizer with BPE-like encoding
•	Gradient descent

## Running an Example

Train a simple MLP using random data:

```bash
go run cmd/examples/neuralnet/main.go
```

It will initialize a network, run forward/backward passes, and print the training loss/accuracy.