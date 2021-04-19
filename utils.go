package tflite

/*
#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>
#cgo LDFLAGS: -ltensorflowlite_c
#cgo linux LDFLAGS: -lm -ldl -lrt
*/
import "C"

// Version returns the TensorFlow Lite version
func Version() string {
	x := C.TfLiteVersion()
	return C.GoString(x)
}
