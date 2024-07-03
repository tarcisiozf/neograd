package engine

import (
	"fmt"
	"math"
)

type ValueList []*Value

func (v ValueList) Item() *Value {
	return v[0]
}

type Value struct {
	data     float32
	prev     []*Value
	grad     float32
	backward func()
	label    string
}

func NewValue(data float32, children ...*Value) *Value {
	return &Value{
		data: data,
		prev: children,
		grad: 0,
	}
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.data+other.data, v, other)
	out.backward = func() {
		v.grad += out.grad
		other.grad += out.grad
	}
	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.data*other.data, v, other)
	out.backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}
	return out
}

// TODO: add ReLU

func (v *Value) Tanh() *Value {
	t := math.Tanh(float64(v.data))
	out := NewValue(float32(t), v)
	out.backward = func() {
		v.grad += float32(1-math.Pow(t, 2)) * out.grad
	}
	return out
}

func (v *Value) Exp() *Value {
	x := float32(math.Exp(float64(v.data)))
	out := NewValue(x, v)
	out.backward = func() {
		v.grad += out.data * out.grad
	}
	return out
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1)) // TODO: optimize
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg()) // TODO: optimize
}

func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1))
}

func (v *Value) Backward() {
	v.grad = 1

	for _, node := range TopologicalSort(v) {
		if node.backward != nil {
			node.backward()
		}
	}
}

func (v *Value) Pow(other float64) *Value {
	x := float32(math.Pow(float64(v.data), other))
	out := NewValue(x, v)
	out.backward = func() {
		v.grad += float32(other*math.Pow(float64(v.data), other-1)) * out.grad
	}
	return out
}

func (v *Value) Data() float32 {
	return v.data
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%f label=%s)", v.data, v.label)
}

func (v *Value) Label(label string) *Value {
	v.label = label
	return v
}

func (v *Value) Grad() float32 {
	return v.grad
}

func (v *Value) Adjust(amount float32) {
	v.data += amount
}

func (v *Value) ZeroGrad() {
	v.grad = 0
}
