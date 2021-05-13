package tflite

/*
#include <stdio.h>
#include <tensorflow/lite/c/c_api.h>
#include <libedgetpu/edgetpu_c.h>
#cgo LDFLAGS: -ltensorflowlite_c -ledgetpu
#cgo linux LDFLAGS: -lm -ldl -lrt
*/
import "C"
import (
	"unsafe"
)

const (
	// Device Types
	TypeApexPCI DeviceType = C.EDGETPU_APEX_PCI
	TypeApexUSB DeviceType = C.EDGETPU_APEX_USB
)

type DeviceType uint32

type Device struct {
	Type DeviceType
	Path string
}

// Delegate represents tflite delegate
type Delegate struct {
	delegate *C.TfLiteDelegate
}

// Return pointer
func (d *Delegate) UsPtr() unsafe.Pointer {
	return unsafe.Pointer(d.delegate)
}
