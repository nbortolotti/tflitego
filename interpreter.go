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

// TfLiteStatus represents TFLiteStatus
type TfLiteStatus int

// options to represent TFLiteStatus
const (
	TfLiteOk TfLiteStatus = iota
	TfLiteError
	TfLiteDelegateError
	TfLiteApplicationError
)

// TfLiteInterpreter represents a  TensorFlow Lite Interpreter .
type TfLiteInterpreter struct {
	interpreter *C.TfLiteInterpreter
}

// NewInterpreter create new TfLiteInterpreter.
func NewInterpreter(tfmodel *TFLiteModel, opts *InterpreterOptions) (*TfLiteInterpreter, error) {
	var o *C.TfLiteInterpreterOptions
	if opts != nil {
		o = opts.options
	}
	i := C.TfLiteInterpreterCreate(tfmodel.model, o)
	if i == nil {
		return nil, fmt.Errorf("unable to create new TFLiteInterpreter")
	}
	return &TfLiteInterpreter{interpreter: i}, nil
}

// Delete represents the delete instance of Interpreter.
func (TfLiteInterpreter *TfLiteInterpreter) Delete() {
	if TfLiteInterpreter != nil {
		C.TfLiteInterpreterDelete(TfLiteInterpreter.interpreter)
	}

}

// GetInputTensor return  tfLiteTensor using index.
func (TfLiteInterpreter *TfLiteInterpreter) GetInputTensor(index int) (*TfLiteTensor, error) {
	t := C.TfLiteInterpreterGetInputTensor(TfLiteInterpreter.interpreter, C.int32_t(index))
	if t == nil {
		return nil, fmt.Errorf("unable to retrieve Input Tensor")
	}
	return &TfLiteTensor{tensor: t}, nil
}

// AllocateTensors allocate tensors for the interpreter.
func (TfLiteInterpreter *TfLiteInterpreter) AllocateTensors() TfLiteStatus {
	if TfLiteInterpreter != nil {
		s := C.TfLiteInterpreterAllocateTensors(TfLiteInterpreter.interpreter)
		return TfLiteStatus(s)
	}
	return TfLiteError
}

// Invoke invoke interpreter
func (TfLiteInterpreter *TfLiteInterpreter) Invoke() TfLiteStatus {
	s := C.TfLiteInterpreterInvoke(TfLiteInterpreter.interpreter)
	return TfLiteStatus(s)
}

// GetOutputTensor return output TfLiteTensor specified by index.
func (TfLiteInterpreter *TfLiteInterpreter) GetOutputTensor(index int) *TfLiteTensor {
	t := C.TfLiteInterpreterGetOutputTensor(TfLiteInterpreter.interpreter, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &TfLiteTensor{tensor: t}
}
