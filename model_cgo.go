package tflite

import (
	"C"
	"reflect"
	"testing"
)

func testNewModelFromFile(t *testing.T) {
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
			input: "test/mobilenet_v2_1.0_224_quant.tflite",
			want: response{
				i: 8,
				e: nil,
			},
		},
		{
			name:  "Emty model",
			input: "",
			want: response{
				i: 0,
				e: ErrCreateModel,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, err := NewModelFromFile(tc.input)
			var got response
			if m == nil {
				got = response{
					i: 0,
					e: err,
				}
			} else {
				got = response{
					i: reflect.TypeOf(m).Size(),
					e: err,
				}
			}

			if got.e != tc.want.e {
				t.Errorf("NewModel got %s = wants %s", got.e, tc.want.e)
			}
		})
	}
}

func testDelete(t *testing.T) {

	type test struct {
		input string
		want  error
	}

	tests := []test{
		{input: "test/mobilenet_v2_1.0_224_quant.tflite", want: nil},
		{input: "", want: ErrDeleteModel},
	}

	for _, tc := range tests {
		m, _ := NewModelFromFile(tc.input)

		got := m.Delete()
		if got != tc.want {
			t.Fatalf("expected: %v, got: %v", tc.want, got)
		}
	}
}
