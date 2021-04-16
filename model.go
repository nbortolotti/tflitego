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
	if path != "" {
		m := C.TfLiteModelCreateFromFile(C.CString(path))
		if m == nil {
			return nil, ErrCreateModel
		}
		return &Model{model: m}, nil
	}
	return nil, ErrCreateModel
}

// Delete delete instance of TF Lite model.
func (m *Model) Delete() error {
	if m != nil {
		C.TfLiteModelDelete(m.model)
		return nil
	}
	return ErrDeleteModel
}
