package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"neograd/zero/matrix"
	"os"
	"strconv"
	"strings"
)

type MemState struct {
	X   [][]float64 `json:"X"`
	Y   []float64   `json:"Y"`
	Z1  [][]float64 `json:"Z1"`
	A1  [][]float64 `json:"A1"`
	Z2  [][]float64 `json:"Z2"`
	A2  [][]float64 `json:"A2"`
	OhY [][]float64 `json:"ohY"`
	DZ2 [][]float64 `json:"dZ2"`
	DW2 [][]float64 `json:"dW2"`
	Db2 float64     `json:"db2"`
	DZ1 [][]float64 `json:"dZ1"`
	DW1 [][]float64 `json:"dW1"`
	Db1 float64     `json:"db1"`
	W1  [][]float64 `json:"W1"`
	B1  [][]float64 `json:"b1"`
	W2  [][]float64 `json:"W2"`
	B2  [][]float64 `json:"b2"`
}

var memState = MemState{}

func main() {
	var X [][]float64
	var Y []float64

	file, err := os.ReadFile("./scratch/train.csv")
	if err != nil {
		panic(err)
	}
	csv := strings.Split(string(file), "\r\n")
	for _, row := range csv[1:] {
		if row == "" {
			continue
		}
		cols := strings.Split(row, ",")
		label, err := strconv.Atoi(cols[0])
		if err != nil {
			panic(err)
		}
		Y = append(Y, float64(label))
		pixels := make([]float64, 784)
		for i, col := range cols[1:] {
			p, err := strconv.Atoi(col)
			if err != nil {
				panic(err)
			}
			pixels[i] = float64(p)
		}
		X = append(X, pixels)
	}

	//convertDataset(X, Y)
	convertPretrained()

	X_train := matrix.FromSlice(X[:1000]).Transpose().Divf(255)
	Y_train := Y[:1000]

	gradientDescent(X_train, Y_train, 10000, 0.1)
	//
	//bar := `[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [188], [255], [94], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [191], [250], [253], [93], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [123], [248], [253], [167], [10], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [80], [247], [253], [208], [13], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [29], [207], [253], [235], [77], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [54], [209], [253], [253], [88], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [93], [254], [253], [238], [170], [17], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [23], [210], [254], [253], [159], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [16], [209], [253], [254], [240], [81], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [27], [253], [253], [254], [13], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [20], [206], [254], [254], [198], [7], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [168], [253], [253], [196], [7], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [20], [203], [253], [248], [76], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [22], [188], [253], [245], [93], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [103], [253], [253], [191], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [89], [240], [253], [195], [25], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [15], [220], [253], [253], [80], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [94], [253], [253], [253], [94], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [89], [251], [253], [250], [131], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [214], [218], [95], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]`
	//var baz [][]float64
	//if err := json.Unmarshal([]byte(bar), &baz); err != nil {
	//	panic(err)
	//}
	//foo := matrix.FromSlice(baz).Divf(255)
	//plotImage(foo)

	//foo := X_train.Col(0)
	//plotImage(foo)
	//w1, b1, w2, b2 := loadParams()
	//_, _, _, a2 := forwardPass(w1, b1, w2, b2, foo2)
	//fmt.Println("prediction", prediction(a2))
	//fmt.Println("actual", Y[0])

	//gradientDescent(foo, []float64{Y[0]}, 100, 0.1)
	gradientDescent(X_train.Cols(10), slice(Y, 10), 1, 0.1)

	dump, _ := json.Marshal(memState)
	_ = os.WriteFile("go-compare.json", dump, 0644)
}

func slice(y []float64, i int) []float64 {
	var out []float64
	for j := 0; j < i; j++ {
		out = append(out, y[j])
	}
	return out
}

func convertPretrained() {
	file, err := os.ReadFile("./zero/params.json")
	if err != nil {
		panic(err)
	}
	var params map[string][][]float64
	if err := json.Unmarshal(file, &params); err != nil {
		panic(err)
	}

	dest, err := os.Create("pretrained.bin")
	if err != nil {
		panic(err)
	}
	defer dest.Close()

	b4 := make([]byte, 4)
	b8 := make([]byte, 8)
	for _, k := range []string{"w1", "b1", "w2", "b2"} {
		v := params[k]

		binary.LittleEndian.PutUint32(b4, uint32(len(v)))
		dest.Write(b4)
		binary.LittleEndian.PutUint32(b4, uint32(len(v[0])))
		dest.Write(b4)

		for _, row := range v {
			for _, val := range row {
				binary.LittleEndian.PutUint64(b8, math.Float64bits(val))
				dest.Write(b8)
			}
		}
	}
}

func convertDataset(pixels [][]float64, labels []float64) {
	if len(pixels) != len(labels) {
		panic("Length of pixels and labels do not match")
	}

	f, err := os.Create("dataset.bin")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	b4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(b4, uint32(len(pixels)))
	f.Write(b4)

	for i := range pixels {
		binary.LittleEndian.PutUint32(b4, uint32(labels[i]))
		f.Write(b4)
		if len(pixels[i]) != 784 {
			panic("Invalid number of pixels")
		}
		bpx := make([]byte, len(pixels[i])*8)
		for j := range pixels[i] {
			binary.LittleEndian.PutUint64(bpx[j*8:], math.Float64bits(pixels[i][j]))
		}
		f.Write(bpx)
	}
}

func loadParams() (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	file, err := os.ReadFile("./zero/params.json")
	if err != nil {
		panic(err)
	}
	var params map[string][][]float64
	if err := json.Unmarshal(file, &params); err != nil {
		panic(err)
	}
	w1 := matrix.FromSlice(params["w1"])
	b1 := matrix.FromSlice(params["b1"])
	w2 := matrix.FromSlice(params["w2"])
	b2 := matrix.FromSlice(params["b2"])
	return w1, b1, w2, b2
}

func plotImage(m *matrix.Matrix) {
	v := m.Internal()
	img := image.NewGray(image.Rect(0, 0, 28, 28))

	// Find the min and max values in the matrix for normalization
	var min, max float64 = v[0][0], v[0][0]
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			if v[i*28+j][0] < min {
				min = v[i*28+j][0]
			}
			if v[i*28+j][0] > max {
				max = v[i*28+j][0]
			}
		}
	}

	// Normalize the matrix and set pixel values in the image
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			//normalized := v[i*28+j][0] * 255
			normalized := (v[i*28+j][0] - min) / (max - min) * 255
			grayValue := uint8(normalized)
			img.SetGray(j, i, color.Gray{Y: grayValue})
		}
	}

	// Save the image to a file
	f, err := os.Create("output.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	if err := png.Encode(f, img); err != nil {
		panic(err)
	}
}

func initParams() (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	w1 := matrix.Random(10, 784)
	b1 := matrix.Random(10, 1)
	w2 := matrix.Random(10, 10)
	b2 := matrix.Random(10, 1)
	return w1, b1, w2, b2
}

func forwardPass(w1, b1, w2, b2, X *matrix.Matrix) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	//v, _ := json.Marshal(X.Internal())
	//fmt.Println("X", string(v))

	z1 := w1.Dot(X).Add(b1)
	a1 := matrix.ReLU(z1)
	z2 := w2.Dot(a1).Add(b2)
	a2 := matrix.Softmax(z2)

	memState.Z1 = z1.Internal()
	memState.A1 = a1.Internal()
	memState.Z2 = z2.Internal()
	memState.A2 = a2.Internal()

	return z1, a1, z2, a2
}

func derivReLU(z *matrix.Matrix) *matrix.Matrix {
	out := matrix.FromShape(z)
	v := z.Internal()
	for i := range v {
		for j := range v[i] {
			if v[i][j] > 0 {
				out.Set(i, j, 1)
			}
		}
	}
	return out
}

func backProp(z1, a1, z2, a2, w2, x *matrix.Matrix, y []float64) (*matrix.Matrix, float64, *matrix.Matrix, float64) {
	m := 42000 // TODO: len(y) + 1000
	ohY := matrix.OneHot(y)
	dz2 := a2.Sub(ohY)
	dw2 := matrix.Mulf(dz2.Dot(a1.Transpose()), 1/float64(m))
	//fmt.Println(dz2.Sumf())
	db2 := dz2.Sumf() * (1 / float64(m))
	dz1 := w2.Transpose().Dot(dz2).Mul(derivReLU(z1))
	dw1 := matrix.Mulf(dz1.Dot(x.Transpose()), 1/float64(m))
	db1 := dz1.Sumf() * (1 / float64(m))

	memState.OhY = ohY.Internal()
	memState.A2 = a2.Internal()
	memState.DZ2 = dz2.Internal()
	memState.DW2 = dw2.Internal()
	memState.Db2 = db2
	memState.DZ1 = dz1.Internal()
	memState.DW1 = dw1.Internal()
	memState.Db1 = db1

	return dw1, db1, dw2, db2
}

func updateParams(w1, b1, w2, b2, dw1 *matrix.Matrix, db1 float64, dw2 *matrix.Matrix, db2 float64, lr float64) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	w1 = w1.Sub(matrix.Mulf(dw1, lr))
	b1 = b1.Subf(db1 * lr)
	w2 = w2.Sub(matrix.Mulf(dw2, lr))
	b2 = b2.Subf(db2 * lr)

	memState.W1 = w1.Internal()
	memState.B1 = b1.Internal()
	memState.W2 = w2.Internal()
	memState.B2 = b2.Internal()

	return w1, b1, w2, b2
}

func gradientDescent(x *matrix.Matrix, y []float64, iterations int, lr float64) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	w1, b1, w2, b2 := initParams()
	//w1, b1, w2, b2 := loadParams()
	memState.X = x.Internal()
	memState.Y = y
	for i := 0; i < iterations; i++ {
		z1, a1, z2, a2 := forwardPass(w1, b1, w2, b2, x)
		dw1, db1, dw2, db2 := backProp(z1, a1, z2, a2, w2, x, y)
		w1, b1, w2, b2 = updateParams(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)
		if i%50 == 0 || i == iterations-1 {
			fmt.Println("Iteration", i)
			fmt.Println("Accuracy", accuracy(prediction(a2), y))
		}
	}
	return w1, b1, w2, b2
}

func accuracy(a []float64, b []float64) float64 {
	correct := 0
	for i := range a {
		if a[i] == b[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(a))
}

func prediction(m *matrix.Matrix) []float64 {
	var out []float64
	v := m.Internal()
	for i := 0; i < len(v[0]); i++ {
		max := math.Inf(-1)
		var idx int
		for j := 0; j < len(v); j++ {
			if v[j][i] > max {
				max = v[j][i]
				idx = j
			}
		}
		out = append(out, float64(idx))
	}
	return out
}
