package engine

type ActivationFunction func(value *Value) *Value

func ReLU(value *Value) *Value {
	return value.ReLU()
}

func Softmax(value *Value) *Value {
	// TODO: implement
	//return value.Softmax()
	return nil
}

func Tanh(value *Value) *Value {
	return value.Tanh()
}
