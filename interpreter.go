package tflite

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

// Status represents TFLiteStatus
type Status int

// list of options to represent TFLiteStatus
const (
	StatusOk Status = iota
	StatusError
	DelegateError
	ApplicationError
)

// TfLiteInterpreter represents a  TensorFlow Lite Interpreter .
type TfLiteInterpreter struct {
	interpreter *C.TfLiteInterpreter
}

// NewInterpreter create new TfLiteInterpreter.
func NewInterpreter(tfmodel *Model, opts *InterpreterOptions) (*TfLiteInterpreter, error) {
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
func (i *TfLiteInterpreter) Delete() error {
	if i != nil {
		_, err := C.TfLiteInterpreterDelete(i.interpreter)
		if err != nil {
			return ErrDeleteIntepreter
		}
		return nil
	}
	return ErrDeleteIntepreter
}

// GetInputTensor return  tfLiteTensor using index.
func (i *TfLiteInterpreter) GetInputTensor(index int) (*Tensor, error) {
	t := C.TfLiteInterpreterGetInputTensor(i.interpreter, C.int32_t(index))
	if t == nil {
		return nil, fmt.Errorf("unable to retrieve Input Tensor")
	}
	return &Tensor{tensor: t}, nil
}

// AllocateTensors allocate tensors for the interpreter.
func (i *TfLiteInterpreter) AllocateTensors() Status {
	if i != nil {
		s := C.TfLiteInterpreterAllocateTensors(i.interpreter)
		return Status(s)
	}
	return StatusError
}

// Invoke invoke interpreter
func (i *TfLiteInterpreter) Invoke() Status {
	s := C.TfLiteInterpreterInvoke(i.interpreter)
	return Status(s)
}

// GetOutputTensor return output Tensor specified by index.
func (i *TfLiteInterpreter) GetOutputTensor(index int) (*Tensor, error) {
	t := C.TfLiteInterpreterGetOutputTensor(i.interpreter, C.int32_t(index))
	if t == nil {
		return nil, fmt.Errorf("unable to retrieve output Tensor")
	}
	return &Tensor{tensor: t}, nil
}
