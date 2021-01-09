package tflitego

/*
#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>
#cgo LDFLAGS: -ltensorflowlite_c
#cgo linux LDFLAGS: -lm -ldl -lrt
*/
import "C"
import (
	"fmt"
)

// TensorType is types of the tensor.
type TensorType int

// Tensors type
const (
	TfLiteNoType TensorType = iota
	TfLiteFloat32
	TfLiteInt32
	TfLiteUInt8
	TfLiteInt64
	TfLiteString
	TfLiteBool
	TfLiteInt16
	TfLiteComplex64
	TfLiteInt8
)

// Tensor represents TensorFlow Lite Tensor.
type Tensor struct {
	tensor *C.TfLiteTensor
}

// Type return TensorType.
func (t *Tensor) Type() TensorType {
	return TensorType(C.TfLiteTensorType(t.tensor))
}

// NumDims return number of dimensions.
func (t *Tensor) NumDims() int {
	return int(C.TfLiteTensorNumDims(t.tensor))
}

// Dim return dimension of the element specified by index.
func (t *Tensor) Dim(index int) int {
	return int(C.TfLiteTensorDim(t.tensor, C.int32_t(index)))
}

// Shape return shape of the tensor.
func (t *Tensor) Shape() []int {
	shape := make([]int, t.NumDims())
	for i := 0; i < t.NumDims(); i++ {
		shape[i] = t.Dim(i)
	}
	return shape
}

// ByteSize return byte size of the tensor.
func (t *Tensor) ByteSize() uint {
	return uint(C.TfLiteTensorByteSize(t.tensor))
}

// Name return name of the tensor.
func (t *Tensor) Name() string {
	return C.GoString(C.TfLiteTensorName(t.tensor))
}

// SetFloat32 sets float32s.
func (t *Tensor) SetFloat32(v []float32) error {
	if t.Type() != TfLiteFloat32 {
		return fmt.Errorf("type error")
	}
	ptr := C.TfLiteTensorData(t.tensor)
	if ptr == nil {
		return fmt.Errorf("bad tensor")
	}
	n := t.ByteSize() / 4
	to := (*((*[1<<29 - 1]float32)(ptr)))[:n]
	copy(to, v)
	return nil
}

// OperateFloat32 returns float32.
func (t *Tensor) OperateFloat32() []float32 {
	if t.Type() != TfLiteFloat32 {
		return nil
	}
	ptr := C.TfLiteTensorData(t.tensor)
	if ptr == nil {
		return nil
	}
	n := t.ByteSize() / 4
	return (*((*[1<<29 - 1]float32)(ptr)))[:n]
}
