package matrix

import (
	"fmt"
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
		b = broadcast(m, b)
	}

	out := New(len(m.s), len(m.s[0]))
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] + b.s[i][j]
		}
	}
	return out
}

func broadcast(m *Matrix, b *Matrix) *Matrix {
	if m.rows == b.rows && m.cols > b.cols {
		out := New(b.rows, m.cols)
		for i := range out.s {
			for j := range out.s[i] {
				out.s[i][j] = b.s[i][0]
			}
		}
		return out
	}
	panic(fmt.Sprintf("Failed to broadcast shapes (%d, %d) and (%d, %d)", m.rows, m.cols, b.rows, b.cols))
}

func (m *Matrix) Internal() [][]float32 {
	return m.s
}

func (m *Matrix) Transpose() *Matrix {
	out := New(m.cols, m.rows)
	for i := range m.s {
		for j := range m.s[i] {
			out.s[j][i] = m.s[i][j]
		}
	}
	return out
}

func (m *Matrix) Sub(b *Matrix) *Matrix {
	if len(m.s) != len(b.s) || len(m.s[0]) != len(b.s[0]) {
		panic("Matrix dimensions must match")
	}

	out := New(len(m.s), len(m.s[0]))
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] - b.s[i][j]
		}
	}
	return out
}

func (m *Matrix) Sum(axys int) *Matrix {
	var out *Matrix
	if axys == 2 {
		out = New(len(m.s), 1)
		for i := range m.s {
			for j := range m.s[i] {
				out.s[i][0] += m.s[i][j]
			}
		}
	} else {
		panic("Not implemented")
	}
	return out
}

func (m *Matrix) Set(y int, x int, v float32) {
	m.s[y][x] = v
}

func (m *Matrix) Mul(b *Matrix) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] * b.s[i][j]
		}
	}
	return out
}

func (m *Matrix) Divf(f float32) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] / f
		}
	}
	return out
}

func (m *Matrix) Col(idx int) *Matrix {
	out := New(m.rows, 1)
	for i := range out.s {
		out.s[i][0] = m.s[i][idx]
	}
	return out
}

func (m *Matrix) Sumf() float32 {
	var sum float32
	for i := range m.s {
		for j := range m.s[i] {
			sum += m.s[i][j]
		}
	}
	return sum
}

func (m *Matrix) Subf(f float32) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] - f
		}
	}
	return out
}

//func (m *Matrix) Size() int {
//	return m.rows * m.cols
//}
//
//func (m *Matrix) Max() float32 {
//	max := float32(math.Inf(-1))
//	for i := range m.s {
//		for j := range m.s[i] {
//			if m.s[i][j] > max {
//				max = m.s[i][j]
//			}
//		}
//	}
//	return max
//}

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
	sum := float32(0)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = float32(math.Exp(float64(m.s[i][j])))
			sum += out.s[i][j]
		}
	}
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] /= sum
		}
	}
	return out
}

func OneHot(Y []float32) *Matrix {
	size := len(Y)
	max := float32(math.Inf(-1))
	for i := range Y {
		if Y[i] > max {
			max = Y[i]
		}
	}
	out := New(size, 10) // int(max)+1) // TODO: ceil?
	for i := range Y {
		out.s[i][int(Y[i])] = 1
	}
	return out.Transpose()
}

func Mulf(m *Matrix, f float32) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] * f
		}
	}
	return out
}
