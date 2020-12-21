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

// TFVersion return TensorFlow Lite version
func TFVersion() (string, error) {
	x := C.TfLiteVersion()
	if x == nil {
		return "", fmt.Errorf("unable to retrieve TensorFlow Lite version")
	}
	return C.GoString(x), nil
}