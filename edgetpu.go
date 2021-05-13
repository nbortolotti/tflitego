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

// New Edgetpu Delegate.
func NewEdge(device Device) *Delegate {
	d := C.edgetpu_create_delegate(uint32(device.Type), C.CString(device.Path), nil, 0)
	if d == nil {
		return nil
	}
	return &Delegate{
		delegate: d,
	}
}

// Delete Edgetpu delegate.
func (d *Delegate) DeleteEdge() {
	C.edgetpu_free_delegate(d.delegate)
}

// EdgeTPU runtime version.
func VersionEdge() (string, error) {
	version := C.edgetpu_version()
	if version == nil {
		return "", ErrGetEdgeVersion
	}
	return C.GoString(version), nil
}

// Verbosity represents edgetpu verbosity.
func Verbosity(v int) {
	C.edgetpu_verbosity(C.int(v))
}

// DeviceList represent the list of devices.
func DeviceList() ([]Device, error) {
	var numDevices C.size_t
	cDevices := C.edgetpu_list_devices(&numDevices)

	if cDevices == nil {
		return nil, nil
	}

	deviceSlice := (*[1024]C.struct_edgetpu_device)(unsafe.Pointer(cDevices))[:numDevices:numDevices]

	var devices []Device
	for i := C.size_t(0); i < numDevices; i++ {
		devices = append(devices, Device{
			Type: DeviceType(deviceSlice[i]._type),
			Path: C.GoString(deviceSlice[i].path),
		})
	}

	C.edgetpu_free_devices(cDevices)

	return devices, nil
}
