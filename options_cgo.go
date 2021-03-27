package tflite

import (
	"C"
	"testing"
)

func testNewInterpreterOptions(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if o == nil && err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
}

func testSetNumThread(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if o == nil && err != nil {
		t.Errorf("cannot initialize interpreter options")
	}

	err = o.SetNumThread(4)
	if err != nil {
		t.Errorf("cannot set number of threads")
	}

}

func testInterpreterOptionsDelete(t *testing.T) {

	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if o == nil && err != nil {
		t.Errorf("cannot initialize interpreter options")
	}

	err = o.Delete()
	if err != nil {
		t.Errorf("interpreter options Delete not success")
	}
}
