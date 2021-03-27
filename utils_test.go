package tflite

import "testing"

func TestTFVersion(t *testing.T) {
	v, err := Version()
	if err != nil {
		t.Errorf("TF version is not responding: %s", err)
	}
	println(v)
}
