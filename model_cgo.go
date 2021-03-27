package tflite

import (
	"C"
	"testing"
)

func testNewModelFromFile(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("model not success")
	}
}

func testDelete(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("model not success")
	}

	err = m.Delete()
	if err != nil {
		t.Errorf("model Delete not success")
	}
}
