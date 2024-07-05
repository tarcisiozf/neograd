package engine

type ValueList []*Value

func (v ValueList) Item() *Value {
	return v[0]
}

func (v ValueList) DataSlice() []float32 {
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i].data
	}
	return out
}

func ToList(list ...float32) ValueList {
	out := make([]*Value, len(list))
	for i, v := range list {
		out[i] = NewValue(v)
	}
	return out
}
