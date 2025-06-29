package matrix

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	s          [][]float64
	rows, cols int
}

func New(rows int, cols int) *Matrix {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
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

func FromSlice(s [][]float64) *Matrix {
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
	if m.rows > b.rows && m.cols == b.cols {
		out := New(m.rows, b.cols)
		for i := range out.s {
			for j := range out.s[i] {
				out.s[i][j] = b.s[0][j]
			}
		}
		return out
	}
	panic(fmt.Sprintf("Failed to broadcast shapes (%d, %d) and (%d, %d)", m.rows, m.cols, b.rows, b.cols))
}

func (m *Matrix) Internal() [][]float64 {
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

//func (m *Matrix) Sum(axys int) *Matrix {
//	var out *Matrix
//	if axys == 2 {
//		out = New(len(m.s), 1)
//		for i := range m.s {
//			for j := range m.s[i] {
//				out.s[i][0] += m.s[i][j]
//			}
//		}
//	} else {
//		panic("Not implemented")
//	}
//	return out
//}

func (m *Matrix) Set(y int, x int, v float64) {
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

func (m *Matrix) Div(b *Matrix) *Matrix {
	if m.rows != b.rows || m.cols != b.cols {
		b = broadcast(m, b)
	}

	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] / b.s[i][j]
		}
	}
	return out
}

func (m *Matrix) Divf(f float64) *Matrix {
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

func (m *Matrix) Cols(num int) *Matrix {
	out := New(m.rows, num)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j]
		}
	}
	return out
}

func (m *Matrix) Sumf() float64 {
	var sum float64
	for i := range m.s {
		for j := range m.s[i] {
			sum += m.s[i][j]
		}
	}
	return sum
}

func (m *Matrix) Subf(f float64) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] - f
		}
	}
	return out
}

func (m *Matrix) Dump() {
	if bytes, err := json.Marshal(m.s); err == nil {
		fmt.Println(string(bytes))
	}
}

func Random(rows int, cols int) *Matrix {
	out := New(rows, cols)
	for i := range out.s {
		out.s[i] = make([]float64, cols)
		for j := range out.s[i] {
			out.s[i][j] = rand.Float64() - 0.5
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
	sum := New(1, m.cols)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = math.Exp(m.s[i][j])
			sum.s[0][j] += out.s[i][j]
		}
	}
	return out.Div(sum)
}

func OneHot(Y []float64) *Matrix {
	size := len(Y)
	max := float64(math.Inf(-1))
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

func Mulf(m *Matrix, f float64) *Matrix {
	out := FromShape(m)
	for i := range out.s {
		for j := range out.s[i] {
			out.s[i][j] = m.s[i][j] * f
		}
	}
	return out
}
