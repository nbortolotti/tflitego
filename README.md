![Go](https://github.com/nbortolotti/gotflite/workflows/Go/badge.svg?branch=main)
[![GoDoc](https://godoc.org/github.com/nbortolotti/tflitego?status.svg)](https://godoc.org/github.com/nbortolotti/tflitego)


# tflitego: TensorFlow Lite for Go
tflitego provide a simple and clear solution to use TensorFlow lite in Go. Our objective is provide a cohesive API, simplicity related to TensorFlow Lite C API connection and maintainability.

## Requeriments
TensorFlow Lite C API. A native shared library target that contains the C API for inference has been provided. Assuming a working bazel configuration, this can be built as follows:

```shell script
bazel build -c opt //tensorflow/lite/c:tensorflowlite_c
```
more details [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c)

Alternativally, if your prefer a simplification to use TensorFlow Lite C API, I prepared a a package here:
* [linux/X86_64, 2.4.0](https://storage.googleapis.com/clitelibrary/ctflitelib_2.4.0.tar.gz). Tested ubuntu 18.04

```shell script
wget https://storage.googleapis.com/clitelibrary/ctflitelib_[version].tar.gz
sudo tar -C /usr/local -xzf ctflitelib_[version].tar.gz
sudo ldconfig
```
Replaces version for the available package. Example:

```shell script
wget https://storage.googleapis.com/clitelibrary/ctflitelib_2.4.0.tar.gz
```
* [raspberrypi_linux/ARMv7, 2.4.0](https://storage.googleapis.com/clitelibrary/ctflitelib_2.4.0_ARMv7.tar.gz)


```shell script
wget https://storage.googleapis.com/clitelibrary/ctflitelib_2.4.0_ARMv7.tar.gz
sudo tar -C /usr/local -xzf ctflitelib_2.4.0_ARMv7.tar.gz
```

## Installation

```shell script
go get github.com/nbortolotti/tflitego
```

## How to use

1. Create the model, here using the method from file.

```go
model, err := tflite.NewTFLiteModelFromFile("iris_lite.tflite")
defer model.Delete()
if err != nil {
    log.Fatal("cannot load model", err)
}
```

2. Set Interpreter options

```go
options, err := tflite.NewInterpreterOptions()
defer options.Delete()
if err != nil {
    log.Fatal("cannot initialize interpreter options", err)
}
options.SetNumThread(4)
```

3. Create Interpreter

```go
interpreter, err := tflite.NewInterpreter(model, options)
defer interpreter.Delete()
if err != nil {
    log.Fatal("cannot create interpreter", err)
}
```

4. Allocate Tensors

```go
status := interpreter.AllocateTensors()
if status != tflite.TfLiteOk {
    log.Println("allocate Tensors failed")
}
```

5. Input Tensor/s

```go
newspecie := []float32{7.9, 3.8, 6.4, 2.0}
input, err := interpreter.GetInputTensor(0)
input.SetFloat32(newspecie)
```

6. Interpreter Invoke 

```go
status = interpreter.Invoke()
if status != tflite.StatusOk {
    log.Println("invoke interpreter failed")
}
```

7. Outputs/Results

```go
output := interpreter.GetOutputTensor(0)
out := output.OperateFloat32()
fmt.Println(topSpecie(out))
```

<img src="https://storage.googleapis.com/tflitego/iris3.gif?raw=true" width="600px">

Also here is possible view examples about tflitego in action:
* [tflite examples](https://github.com/nbortolotti/tflitego_examples)
