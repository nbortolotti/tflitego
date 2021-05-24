package tflite

import (
	"testing"
)

func TestNewInterpreterOptions(t *testing.T) {
	testNewInterpreterOptions(t)
}

func TestInterpreterOptionsDelete(t *testing.T) {
	testInterpreterOptionsDelete(t)
}

func TestSetNumThread(t *testing.T){
	testSetNumThread(t)
}

func TestAddDelegate(t *testing.T){
	testAddDelegate(t)
}