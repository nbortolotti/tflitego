package tflite

/*
#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>
#cgo LDFLAGS: -ltensorflowlite_c
#cgo linux LDFLAGS: -lm -ldl -lrt
*/
import "C"

// Model represents a TensorFlow Lite Model.
type Model struct {
	model *C.TfLiteModel
}

// NewModelFromFile creates a new TensorFlow Lite Model from File.
func NewModelFromFile(path string) (*Model, error) {
	m := C.TfLiteModelCreateFromFile(C.CString(path))
	if m == nil {
		return nil, ErrCreateModel
	}
	return &Model{model: m}, nil
}

// Delete delete instance of TF Lite model.
func (m *Model) Delete() {
	if m != nil {
		C.TfLiteModelDelete(m.model)
	}
}
