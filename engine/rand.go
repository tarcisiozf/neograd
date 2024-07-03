package engine

import "math/rand"

func RandomUniform(min, max float32) float32 {
	return rand.Float32()*(max-min) + min
}
