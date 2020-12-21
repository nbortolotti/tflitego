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

// TfLiteTensor represents TensorFlow Lite Tensor.
type TfLiteTensor struct {
	tensor *C.TfLiteTensor
}

// Type return TensorType.
func (TfLiteTensor *TfLiteTensor) Type() TensorType {
	return TensorType(C.TfLiteTensorType(TfLiteTensor.tensor))
}

// NumDims return number of dimensions.
func (TfLiteTensor *TfLiteTensor) NumDims() int {
	return int(C.TfLiteTensorNumDims(TfLiteTensor.tensor))
}

// Dim return dimension of the element specified by index.
func (TfLiteTensor *TfLiteTensor) Dim(index int) int {
	return int(C.TfLiteTensorDim(TfLiteTensor.tensor, C.int32_t(index)))
}

// Shape return shape of the tensor.
func (TfLiteTensor *TfLiteTensor) Shape() []int {
	shape := make([]int, TfLiteTensor.NumDims())
	for i := 0; i < TfLiteTensor.NumDims(); i++ {
		shape[i] = TfLiteTensor.Dim(i)
	}
	return shape
}

// ByteSize return byte size of the tensor.
func (TfLiteTensor *TfLiteTensor) ByteSize() uint {
	return uint(C.TfLiteTensorByteSize(TfLiteTensor.tensor))
}

// Name return name of the tensor.
func (TfLiteTensor *TfLiteTensor) Name() string {
	return C.GoString(C.TfLiteTensorName(TfLiteTensor.tensor))
}

// SetFloat32 sets float32s.
func (TfLiteTensor *TfLiteTensor) SetFloat32(v []float32) error {
	if TfLiteTensor.Type() != TfLiteFloat32 {
		return fmt.Errorf("type error")
	}
	ptr := C.TfLiteTensorData(TfLiteTensor.tensor)
	if ptr == nil {
		return fmt.Errorf("bad tensor")
	}
	n := TfLiteTensor.ByteSize() / 4
	to := (*((*[1<<29 - 1]float32)(ptr)))[:n]
	copy(to, v)
	return nil
}

// OperateFloat32 returns float32.
func (TfLiteTensor *TfLiteTensor) OperateFloat32() []float32 {
	if TfLiteTensor.Type() != TfLiteFloat32 {
		return nil
	}
	ptr := C.TfLiteTensorData(TfLiteTensor.tensor)
	if ptr == nil {
		return nil
	}
	n := TfLiteTensor.ByteSize() / 4
	return (*((*[1<<29 - 1]float32)(ptr)))[:n]
}
