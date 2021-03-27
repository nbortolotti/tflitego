package tflite

import (
	"testing"
)

func TestNewInterpreter(t *testing.T) {
	testNewInterpreter(t)
}

func TestInterpreterDelete(t *testing.T) {
	testInterpreterDelete(t)
}

func TestGetInputTensor(t *testing.T) {
	testGetInputTensor(t)
}

func TestGetOutputTensor(t *testing.T) {
	testGetOutputTensor(t)
}
