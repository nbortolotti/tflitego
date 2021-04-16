package tflite

import (
	"C"
	"reflect"
	"testing"
)

func testNewInterpreterOptions(t *testing.T) {
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
			name:  "New interpreter option with empty model",
			input: "",
			want: response{
				i: 8,
				e: nil,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			io, err := NewInterpreterOptions()

			var got response
			if io == nil {
				got = response{
					i: 0,
					e: err,
				}
			} else {
				got = response{
					i: reflect.TypeOf(io).Size(),
					e: err,
				}
			}

			if got.e != tc.want.e {
				t.Errorf("NewInterpreterOption got %s = wants %s", got.e, tc.want.e)
			}
		})
	}
}

func testSetNumThread(t *testing.T) {
	type response struct {
		e error
	}
	tests := []struct {
		name  string
		input int
		want  response
	}{
		{
			name:  "Success SetNumThreads",
			input: 1,
			want: response{
				e: nil,
			},
		},
		{
			name:  "Issues with SetNumThreads",
			input: 1,
			want: response{
				e: ErrInterpreterSetNumThread,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var io *InterpreterOptions
			var err error
			if tc.name == "Success SetNumThreads" {
				io, _ = NewInterpreterOptions()
				err = io.SetNumThread(tc.input)
			} else {
				err = io.SetNumThread(tc.input)
			}

			var got response
			if io == nil {
				got = response{
					e: err,
				}
			}
			if got.e != tc.want.e {
				t.Errorf("InterpreterSetNumThread got %s = wants %s", got.e, tc.want.e)
			}
		})
	}
}

func testInterpreterOptionsDelete(t *testing.T) {
	type response struct {
		e error
	}
	tests := []struct {
		name string
		want response
	}{
		{
			name: "Success Delete InterpreterOption",
			want: response{
				e: nil,
			},
		},
		{
			name: "Issues to Delete InterpreterOption",
			want: response{
				e: ErrDeleteIntepreterOptions,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var io *InterpreterOptions
			var err error
			if tc.name == "Success Delete InterpreterOption" {
				io, _ = NewInterpreterOptions()
				err = io.Delete()
			} else {
				err = io.Delete()
			}

			var got response
			if io == nil {
				got = response{
					e: err,
				}
			}
			if got.e != tc.want.e {
				t.Errorf("Delete InterpreterOption got %s = wants %s", got.e, tc.want.e)
			}
		})
	}
}
