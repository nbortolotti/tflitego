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
	// ErrCreateModel is returned when Tensor Flow Lite fails to delete the model
	ErrDeleteModel = errors.New("unable to delete the TF model")
	// ErrDeleteIntepreter is returned when Tensor Flow Lite fails to delete the model interpreter
	ErrDeleteIntepreter = errors.New("unable to delete the TF model interpreter")
	// ErrDeleteIntepreteroptions is returned when Tensor Flow Lite fails to delete the model interpreter options
	ErrDeleteIntepreterOptions = errors.New("unable to delete the TF model interpreter options")
	// ErrInterpreterSetNumThread is returned when Tensor Flow Lite fails to set number of threads
	ErrInterpreterSetNumThread = errors.New("unable to set number of threads")
)
