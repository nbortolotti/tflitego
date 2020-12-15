package gotflite

/*
#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>
#cgo LDFLAGS: -ltensorflowlite_c
#cgo linux LDFLAGS: -lm -ldl -lrt
*/
import "C"

// TFVersion ....
func TFVersion() (string, error) {
	x := C.TfLiteVersion()
	return C.GoString(x), nil
}
