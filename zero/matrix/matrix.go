package matrix

import (
	"math"
	"math/rand"
)

type Matrix struct {
	s          [][]float32
	rows, cols int
}

func New(rows int, cols int) *Matrix {
	out := make([][]float32, rows)
	for i := range out {
		out[i] = make([]float32, cols)
	}
	return &Matrix{
		s:    out,
		rows: rows,
		cols: cols,
	}
}

func FromShape(m *Matrix) *Matrix {
	return New(m.rows, m.cols)
}

func FromSlice(s [][]float32) *Matrix {
	return &Matrix{
		s:    s,
		rows: len(s),
		cols: len(s[0]),
	}
}

func (m *Matrix) Dot(b *Matrix) *Matrix {
	if len(m.s[0]) != len(b.s) {
		panic("Matrix dimensions must match")
	}

	out := New(len(m.s), len(b.s[0]))
	for i := range out.s {
		for j := range out.s[i] {
			for k := range m.s[i] {
				out.s[i][j] += m.s[i][k] * b.s[k][j]
			}
		}
	}
	return out
}

func (m *Matrix) Add(b *Matrix) *Matrix {
	if len(m.s) != len(b.s) || len(m.s[0]) != len(b.s[0]) {
		panic("Matrix dimensions must match")
	}

	out := New(len(m.s), len(m.s[0]))
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] + b.s[i][j]
		}
	}
	return out
}

func (m *Matrix) Internal() [][]float32 {
	return m.s
}

func Random(rows int, cols int) *Matrix {
	out := New(rows, cols)
	for i := range out.s {
		out.s[i] = make([]float32, cols)
		for j := range out.s[i] {
			out.s[i][j] = rand.Float32() - 0.5
		}
	}
	return out
}

func ReLU(m *Matrix) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			if m.s[i][j] > 0 {
				out.s[i][j] = m.s[i][j]
			}
		}
	}
	return out
}

func Softmax(m *Matrix) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		sum := float32(0)
		for j := range out.s[i] {
			out.s[i][j] = float32(math.Exp(float64(m.s[i][j])))
			sum += out.s[i][j]
		}
		for j := range out.s[i] {
			out.s[i][j] /= sum
		}
	}
	return out
}
