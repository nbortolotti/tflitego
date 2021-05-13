package tflite

import (
	"errors"
)

var (
	// ErrGetVersion is returned when Tensor Flow Lite fails to return the version.
	ErrGetVersion = errors.New("unable to retrieve TensorFlow Lite version")
	// ErrGetEdgeVersion is returned when Tensor Flow Lite Edge support fails to return the version.
	ErrGetEdgeVersion = errors.New("unable to retrieve TensorFlow Lite Edge version")
	// ErrCreateModel is returned when Tensor Flow Lite fails to create the model.
	ErrCreateModel = errors.New("unable to create the TF model")
	// ErrCreateInterpreterOptions is returned when creating the interpreter options fails.
	ErrCreateInterpreterOptions = errors.New("unable to create an interpreter options")
	// ErrCreateModel is returned when Tensor Flow Lite fails to delete the model.
	ErrDeleteModel = errors.New("unable to delete the TF model")
	//ErrCreateIntepreter is returned when Tensor Flow Lite fails to create the model interpreter
	ErrCreateIntepreter = errors.New("unable to create the TF model interpreter")
	// ErrDeleteIntepreter is returned when Tensor Flow Lite fails to delete the model interpreter.
	ErrDeleteIntepreter = errors.New("unable to delete the TF model interpreter")
	// ErrDeleteIntepreteroptions is returned when Tensor Flow Lite fails to delete the model interpreter options.
	ErrDeleteIntepreterOptions = errors.New("unable to delete the TF model interpreter options")
	// ErrInterpreterSetNumThread is returned when Tensor Flow Lite fails to set number of threads.
	ErrInterpreterSetNumThread = errors.New("unable to set number of threads")
	//ErrInputTensor is returned when TensorFlow Lite fails to get input tensor
	ErrInputTensor = errors.New("unable to retrieve input Tensor")
	//ErrOutputTensor is returned when TensorFlow Lite fails to get output tensor.
	ErrOutputTensor = errors.New("unable to retrieve output Tensor")
	//ErrType is returned when and image operations fails a get an issue with the image type.
	ErrType = errors.New("unable to set image type")
	//ErrDecode is returned when and decode image operations fails.
	ErrDecode = errors.New("unable to decode image")
)
