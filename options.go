package tflite

/*
#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>
#cgo LDFLAGS: -ltensorflowlite_c
#cgo linux LDFLAGS: -lm -ldl -lrt
*/
import "C"

// InterpreterOptions represents a TensorFlow Lite InterpreterOptions.
type InterpreterOptions struct {
	options *C.TfLiteInterpreterOptions
}

// NewInterpreterOptions creates new InterpreterOptions.
func NewInterpreterOptions() (*InterpreterOptions, error) {
	o := C.TfLiteInterpreterOptionsCreate()
	if o == nil {
		return nil, ErrCreateInterpreterOptions
	}
	return &InterpreterOptions{options: o}, nil
}

// Delete delete instance of InterpreterOptions.
func (o *InterpreterOptions) Delete() error {
	if o != nil {
		_, err := C.TfLiteInterpreterOptionsDelete(o.options)
		if err != nil {
			return ErrDeleteIntepreterOptions
		}
		return nil
	}
	return ErrDeleteIntepreterOptions
}

// SetNumThread set number of threads.
func (o *InterpreterOptions) SetNumThread(numThreads int) error {
	_, err := C.TfLiteInterpreterOptionsSetNumThreads(o.options, C.int32_t(numThreads))
	if err != nil {
		return ErrInterpreterSetNumThread
	}
	return nil
}
