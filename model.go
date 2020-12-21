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

// TFLiteModel represents a TensorFlow Lite Model.
type TFLiteModel struct {
	model *C.TfLiteModel
}

// NewTFLiteModelFromFile creates a new TensorFlow Lite Model from File.
func NewTFLiteModelFromFile(modelPath string) (*TFLiteModel, error) {
	m := C.TfLiteModelCreateFromFile(C.CString(modelPath))
	if m == nil {
		return nil, fmt.Errorf("unable to create the model")
	}
	return &TFLiteModel{model: m}, nil
}

// Delete delete instance of TF Lite model.
func (TFLiteModel *TFLiteModel) Delete() {
	if TFLiteModel != nil {
		C.TfLiteModelDelete(TFLiteModel.model)
	}
}
