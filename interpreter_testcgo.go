package tflite

import (
	"C"
	"reflect"
	"testing"
)

func testNewInterpreter(t *testing.T) {
	type response struct {
		i uintptr
		e error
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "Quant model",
			input: "testing/mobilenet_v2_1.0_224_quant.tflite",
			want: response{
				i: 8,
				e: nil,
			},
		},
		{
			name:  "Interpreter Emty model",
			input: "",
			want: response{
				i: 0,
				e: ErrCreateIntepreter,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			it, err := NewInterpreter(m, o)

			var got response
			if it == nil {
				got = response{
					i: 0,
					e: err,
				}
			} else {
				got = response{
					i: reflect.TypeOf(it).Size(),
					e: err,
				}
			}

			if got.e != tc.want.e {
				t.Errorf("NewInterpreter got %s = wants %s", got.e, tc.want.e)
			}
		})
	}
}

func testInterpreterDelete(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  error
	}{
		{
			name:  "quant model",
			input: "testing/mobilenet_v2_1.0_224_quant.tflite",
			want:  nil,
		},
		{
			name:  "empty model",
			input: "",
			want:  ErrDeleteIntepreter,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()

			i, _ := NewInterpreter(m, o)

			got := i.Delete()
			if got != tc.want {
				t.Fatalf("expected: %v, got: %v", tc.want, got)
			}
		})
	}
}

func testGetInputTensor(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  error
	}{
		{
			name:  "General model",
			input: "testing/iris_lite.tflite",
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			m, _ := NewModelFromFile("testing/mobilenet_v2_1.0_224_quant.tflite")

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)

			if _, got := i.GetInputTensor(0); got != tc.want {
				t.Errorf("GetInputTensor got %s = wants %s", got, tc.want)
			}
		})
	}
}

func testGetOutputTensor(t *testing.T) {
	type response struct {
		tb uint
		e  error
	}

	tests := []struct {
		name  string
		input string
		want  response
	}{
		{
			name:  "General model",
			input: "testing/iris_lite.tflite",
			want: response{
				tb: 12,
				e:  nil,
			},
		},
		{
			name:  "General model",
			input: "",
			want: response{
				tb: 0,
				e:  ErrOutputTensor,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := NewModelFromFile(tc.input)

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)

			i, _ := NewInterpreter(m, o)

			i.AllocateTensors()

			input, _ := i.GetInputTensor(0)

			ts := []float32{7.9, 3.8, 6.4, 2.0}
			_ = input.SetFloat32(ts)

			i.Invoke()
			tensor, err := i.GetOutputTensor(0)

			var got response
			if tensor == nil {
				got = response{
					tb: 0,
					e:  err,
				}
			} else {
				got = response{
					tb: tensor.ByteSize(),
					e:  err,
				}
			}

			if got != tc.want {
				t.Errorf("GetOutputTensor got %s = wants %s", got.e, tc.want.e)
			}
		})
	}
}
