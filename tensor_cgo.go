package tflite

import (
	"C"
	"fmt"
	"image"
	_ "image/png"
	"os"
	"testing"
	"unsafe"

	"github.com/nfnt/resize"
)

func testNumDims(t *testing.T) {
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

	want := 2
	got := input.NumDims()
	if got != want {
		t.Errorf("Got %c but wants %c", got, want)
	}
}

func testByteSize(t *testing.T) {
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

	var want uint = 16
	got := input.ByteSize()
	if got != want {
		t.Errorf("Got %c but wants %c", got, want)
	}
}

func testShape(t *testing.T) {
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

	want := []int{1, 4}
	got := input.Shape()
	if len(got) != len(want) {
		t.Errorf("Got %c but wants %c", got, want)
	}
}

func testName(t *testing.T) {
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

	want := "dense_input"
	got := input.Name()
	if got != want {
		t.Errorf("Got %s but wants %s", got, want)
	}
}

func testData(t *testing.T) {
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

	want := 8
	if unsafe.Sizeof(input.Data()) != uintptr(want) {
		t.Errorf("got %v but wants %v", unsafe.Sizeof(input.Data()), uintptr(want))
	}
}

func testSetFloat32(t *testing.T) {
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
}

func testOperateFloat32(t *testing.T) {
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

	want := []float32{7.9, 3.8, 6.4, 2.0}

	err = input.SetFloat32(want)
	if err != nil {
		t.Errorf("cannot Set Tensor: %s", err)
	}

	got := input.OperateFloat32()
	if len(got) != len(want) {
		t.Errorf("Got %c but wants %c", len(got), len(want))
	}

}

func testFromBuffer(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("model not success")
	}

	o, err := NewInterpreterOptions()
	if err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
	o.SetNumThread(1)

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

	ibuffer, err := imageToBuffer("test/cat.png", input)
	if err != nil {
		t.Errorf("cannot transform image to buffer")
	}

	want := StatusOk
	got := input.FromBuffer(ibuffer)
	if got != want {
		t.Errorf("got %c but wants %c", got, want)
	}
}

func testToBuffer(t *testing.T) {
	m, err := NewModelFromFile("test/mobilenet_v2_1.0_224_quant.tflite")
	if m == nil && err != nil {
		t.Errorf("model not success")
	}

	o, err := NewInterpreterOptions()
	if err != nil {
		t.Errorf("cannot initialize interpreter options")
	}
	o.SetNumThread(1)

	i, err := NewInterpreter(m, o)
	if i == nil && err != nil {
		t.Errorf("cannot create interpreter")
	}

	sa := i.AllocateTensors()
	if sa != StatusOk {
		t.Errorf("allocate Tensors failed")
	}

	input, err := i.GetInputTensor(0)
	if input == nil && err != nil {
		t.Errorf("cannot Get Input Tensor")
	}

	ibuffer, err := imageToBuffer("test/cat.png", input)
	if err != nil {
		t.Errorf("cannot transform image to buffer")
	}

	sfb := input.FromBuffer(ibuffer)
	if sfb != StatusOk {
		t.Errorf("from buffer failed")
	}

	si := i.Invoke()
	if si != StatusOk {
		t.Errorf("invoke failed")
	}

	output, err := i.GetOutputTensor(0)
	if err != nil {
		t.Errorf("cannot get output Tensor")
	}

	outputSize := output.Dim(output.NumDims() - 1)
	b := make([]byte, outputSize)

	want := StatusOk
	got := output.ToBuffer(&b[0])
	if got != want {
		t.Errorf("got %c but wants %c", got, want)
	}

}

func imageToBuffer(imagePath string, t *Tensor) ([]byte, error) {
	imageHeight := t.Dim(1)
	imagewidth := t.Dim(2)
	channels := t.Dim(3)
	wantedType := t.Type()

	img, err := decodeImage(imagePath)
	if err != nil {
		return nil, fmt.Errorf("incorrect image decode")
	}

	resized := resize.Resize(uint(imagewidth), uint(imageHeight), img, resize.NearestNeighbor)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	if wantedType == TfLiteUInt8 {
		bb := make([]byte, dx*dy*channels)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				i := y*dx + x
				bb[(i)*3+0] = byte(float64(r) / 255.0)
				bb[(i)*3+1] = byte(float64(g) / 255.0)
				bb[(i)*3+2] = byte(float64(b) / 255.0)
			}
		}
		return bb, nil
	}
	return nil, fmt.Errorf("incorrect type")
}

func decodeImage(imagePath string) (image.Image, error) {
	f, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("incorrect type")
	}

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode issue")
	}
	return img, nil
}