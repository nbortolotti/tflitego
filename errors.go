package tflite

import (
	"errors"
)

var (
	// ErrGetVersion is returned when Tensor Flow Lite fails to return the version.
	ErrGetVersion = errors.New("unable to retrieve TensorFlow Lite version")
	// ErrCreateModel is returned when Tensor Flow Lite fails to create the model.
	ErrCreateModel = errors.New("unable to create the TF model")
	// ErrCreateInterpreterOptions is returned when creating the interpreter options fails.
	ErrCreateInterpreterOptions = errors.New("unable to create an interpreter options")
)
