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
	type response struct {
		nd int
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "Tensor number of dimensions with a general model",
			input: "testing/iris_lite.tflite",
			want: response{
				nd: 2,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)
			_ = i.AllocateTensors()

			input, err := i.GetInputTensor(0)
			if input == nil && err != nil {
				t.Errorf("cannot get input Tensor")
			}

			ts := []float32{7.9, 3.8, 6.4, 2.0}

			err = input.SetFloat32(ts)
			if err != nil {
				t.Errorf("cannot set Tensor: %s", err)
			}

			got := input.NumDims()
			if got != tc.want.nd {
				t.Errorf("expected: %v, got: %v", tc.want.nd, got)
			}

		})
	}
}

func testByteSize(t *testing.T) {
	type response struct {
		bs uint
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "Tensor's bytes size with a general model",
			input: "testing/iris_lite.tflite",
			want: response{
				bs: 16,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)
			_ = i.AllocateTensors()

			input, err := i.GetInputTensor(0)
			if input == nil && err != nil {
				t.Errorf("cannot get input Tensor")
			}

			ts := []float32{7.9, 3.8, 6.4, 2.0}

			err = input.SetFloat32(ts)
			if err != nil {
				t.Errorf("cannot Set Tensor: %s", err)
			}

			got := input.ByteSize()
			if got != tc.want.bs {
				t.Errorf("expected: %v, got: %v", tc.want.bs, got)
			}

		})
	}
}

func testShape(t *testing.T) {
	type response struct {
		ts []int
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "Tensor's shape with a general model",
			input: "testing/iris_lite.tflite",
			want: response{
				ts: []int{1, 4},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)
			_ = i.AllocateTensors()

			input, err := i.GetInputTensor(0)
			if input == nil && err != nil {
				t.Errorf("cannot get input Tensor")
			}

			ts := []float32{7.9, 3.8, 6.4, 2.0}

			err = input.SetFloat32(ts)
			if err != nil {
				t.Errorf("cannot Set Tensor: %s", err)
			}

			got := input.Shape()
			if len(got) != len(tc.want.ts) {
				t.Fatalf("expected: %v, got: %v", len(tc.want.ts), len(got))
			}

		})
	}
}

func testName(t *testing.T) {
	type response struct {
		n string
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "Tensor name with a general model",
			input: "testing/iris_lite.tflite",
			want: response{
				n: "dense_input",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)
			_ = i.AllocateTensors()

			input, err := i.GetInputTensor(0)
			if input == nil && err != nil {
				t.Errorf("cannot get input Tensor")
			}

			ts := []float32{7.9, 3.8, 6.4, 2.0}

			err = input.SetFloat32(ts)
			if err != nil {
				t.Errorf("cannot set Tensor: %s", err)
			}

			got := input.Name()
			if got != tc.want.n {
				t.Errorf("expected: %v, got: %v", tc.want.n, got)
			}
		})
	}
}

func testData(t *testing.T) {
	type response struct {
		d uintptr
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "Tensor data with a general model",
			input: "testing/iris_lite.tflite",
			want: response{
				d: 8,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)
			_ = i.AllocateTensors()

			input, err := i.GetInputTensor(0)
			if input == nil && err != nil {
				t.Errorf("cannot get input Tensor")
			}

			ts := []float32{7.9, 3.8, 6.4, 2.0}

			err = input.SetFloat32(ts)
			if err != nil {
				t.Errorf("cannot set Tensor: %s", err)
			}

			got := unsafe.Sizeof(input.Data())
			if got != tc.want.d {
				t.Errorf("expected: %v, got: %v", tc.want.d, got)
			}
		})
	}
}

func testSetFloat32(t *testing.T) {
	m, _ := NewModelFromFile("testing/iris_lite.tflite")

	o, _ := NewInterpreterOptions()
	o.SetNumThread(4)

	i, _ := NewInterpreter(m, o)

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
	m, _ := NewModelFromFile("testing/iris_lite.tflite")

	o, _ := NewInterpreterOptions()
	o.SetNumThread(4)

	i, _ := NewInterpreter(m, o)

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
	m, _ := NewModelFromFile("testing/mobilenet_v2_1.0_224_quant.tflite")

	o, _ := NewInterpreterOptions()
	o.SetNumThread(1)

	i, _ := NewInterpreter(m, o)

	s := i.AllocateTensors()
	if s != StatusOk {
		t.Errorf("allocate Tensors failed")
	}

	input, _ := i.GetInputTensor(0)

	ibuffer, _ := imageToBuffer("testing/cat.png", input)

	want := StatusOk
	got := input.FromBuffer(ibuffer)
	if got != want {
		t.Errorf("got %c but wants %c", got, want)
	}
}

func testToBuffer(t *testing.T) {
	m, _ := NewModelFromFile("testing/mobilenet_v2_1.0_224_quant.tflite")

	o, _ := NewInterpreterOptions()
	o.SetNumThread(1)

	i, _ := NewInterpreter(m, o)

	sa := i.AllocateTensors()
	if sa != StatusOk {
		t.Errorf("allocate Tensors failed")
	}

	input, _ := i.GetInputTensor(0)

	ibuffer, err := imageToBuffer("testing/cat.png", input)
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

	output, _ := i.GetOutputTensor(0)

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
