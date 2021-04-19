package tflite

import (
	"testing"
)

func TestNumDims(t *testing.T) {
	testNumDims(t)
}

func TestByteSize(t *testing.T) {
	testByteSize(t)
}

func TestShape(t *testing.T) {
	testShape(t)
}

func TestName(t *testing.T) {
	testName(t)
}

func TestSetFloat32(t *testing.T) {
	testSetFloat32(t)
}

func TestOperateFloat32(t *testing.T) {
	testOperateFloat32(t)
}

func TestFromBuffer(t *testing.T) {
	testFromBuffer(t)
}

func TestToBuffer(t *testing.T) {
	testToBuffer(t)
}

func TestDecodeImage(t *testing.T){
	testDecodeImage(t)
}