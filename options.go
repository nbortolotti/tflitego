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
		C.TfLiteInterpreterOptionsDelete(o.options)
		return nil
	}
	return ErrDeleteIntepreterOptions
}

// SetNumThread set number of threads.
func (o *InterpreterOptions) SetNumThread(numThreads int) error {
	if o != nil {
		C.TfLiteInterpreterOptionsSetNumThreads(o.options, C.int32_t(numThreads))
		return nil
	}
	return ErrInterpreterSetNumThread
}

// AddDelegate add the option of a delegate for the TensorFlow Lite interpreter
func (o *InterpreterOptions) AddDelegate(d *Delegate) {
	C.TfLiteInterpreterOptionsAddDelegate(o.options, (*C.TfLiteDelegate)(d.UsPtr()))
}
