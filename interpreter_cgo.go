package tflite

import (
	"C"
	"testing"
)

func testNewInterpreter(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
	o.SetNumThread(4)

	i, err := NewInterpreter(m, o)
	if i == nil && err != nil {
		t.Errorf("cannot create interpreter")
	}
}

func testInterpreterDelete(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
	o.SetNumThread(4)

	i, err := NewInterpreter(m, o)
	if i == nil && err != nil {
		t.Errorf("cannot create interpreter")
	}

	err = i.Delete()
	if err != nil {
		t.Errorf("interpreter Delete not success")
	}
}

func testGetInputTensor(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
	o.SetNumThread(4)

	i, err := NewInterpreter(m, o)
	if i == nil && err != nil {
		t.Errorf("cannot create interpreter")
	}

	input, err := i.GetInputTensor(0)
	if input == nil && err != nil {
		t.Errorf("cannot Get Input Tensor")
	}
}

func testGetOutputTensor(t *testing.T) {
	m, err := NewModelFromFile("test/iris_lite.tflite")
	if m == nil && err != nil {
		t.Errorf("Model not success")
	}

	o, err := NewInterpreterOptions()
	if err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
	o.SetNumThread(4)

	i, err := NewInterpreter(m, o)
	if i == nil && err != nil {
		t.Errorf("cannot create interpreter")
	}

	s := i.AllocateTensors()
	if s != StatusOk {
		t.Errorf("allocate Tensors failed")
	}

	input, err := i.GetInputTensor(0)
	if input == nil && err != nil {
		t.Errorf("cannot Get Input Tensor")
	}

	ts := []float32{7.9, 3.8, 6.4, 2.0}

	err = input.SetFloat32(ts)
	if err != nil {
		t.Errorf("cannot Set Tensor: %s", err)
	}

	ivs := i.Invoke()
	if ivs != StatusOk {
		t.Errorf("invoke interpreter failed")
	}

	_, err = i.GetOutputTensor(0)
	if err != nil {
		t.Errorf("invoke interpreter failed")
	}
}
