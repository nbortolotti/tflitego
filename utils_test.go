package tflitego

import "testing"

func TestTFVersion(t *testing.T) {
	v, err := TFVersion()
	if err != nil {
		t.Errorf("TF version is not responding")
	}
	println(v)
}
