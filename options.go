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

// InterpreterOptions represents a TensorFlow Lite InterpreterOptions.
type InterpreterOptions struct {
	options *C.TfLiteInterpreterOptions
}

// NewInterpreterOptions creates new InterpreterOptions.
func NewInterpreterOptions() (*InterpreterOptions, error) {
	o := C.TfLiteInterpreterOptionsCreate()
	if o == nil {
		return nil, fmt.Errorf("unable to create an InterpreterOptions")
	}
	return &InterpreterOptions{options: o}, nil
}

// Delete delete instance of InterpreterOptions.
func (InterpreterOptions *InterpreterOptions) Delete() {
	if InterpreterOptions != nil {
		C.TfLiteInterpreterOptionsDelete(InterpreterOptions.options)
	}
}

// SetNumThread set number of threads.
func (InterpreterOptions *InterpreterOptions) SetNumThread(numThreads int) {
	C.TfLiteInterpreterOptionsSetNumThreads(InterpreterOptions.options, C.int32_t(numThreads))
}
