package tflite

import (
	"C"
	"testing"
)

func testNewEdge(t *testing.T) {
	type response struct {
		d *Delegate
	}

	tests := []struct {
		name      string
		input     string
		imagePath string
		want      response
	}{
		{
			name:      "New Edge",
			input:     "testing/mobilenet_v2_1.0_224_quant.tflite",
			imagePath: "testing/cat.png",
			want: response{
				d: nil,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			m, _ := NewModelFromFile(tc.input)

			d := NewEdge(Device{
				Type: TypeApexUSB,
				Path: "",
			})

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)
			_, _ = NewInterpreter(m, o)

			got := d
			if got != tc.want.d {
				t.Errorf("Got %v but wants %v", got, tc.want.d)
			}
		})
	}
}

func testDeleteEdge(t *testing.T) {
	type response struct {
		er error
	}

	tests := []struct {
		name      string
		input     string
		imagePath string
		want      response
	}{
		{
			name:      "Delete delegate edge err",
			input:     "testing/mobilenet_v2_1.0_224_quant.tflite",
			imagePath: "testing/cat.png",
			want: response{
				er: ErrDeleteDelegateEdge,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			m, _ := NewModelFromFile(tc.input)

			d := NewEdge(Device{
				Type: TypeApexUSB,
				Path: "",
			})

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)
			_, _ = NewInterpreter(m, o)
			err := d.DeleteEdge()

			got := err
			if got != tc.want.er {
				t.Errorf("Got %v but wants %v", got, tc.want.er)
			}
		})
	}
}

func testVersionEdge(t *testing.T) {
	type response struct {
		v  string
		er error
	}

	tests := []struct {
		name      string
		input     string
		imagePath string
		want      response
	}{
		{
			name:      "General installed edge version",
			input:     "testing/mobilenet_v2_1.0_224_quant.tflite",
			imagePath: "testing/cat.png",
			want: response{
				v:  "",
				er: nil,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			m, _ := NewModelFromFile(tc.input)

			_ = NewEdge(Device{
				Type: TypeApexUSB,
				Path: "",
			})

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)
			_, _ = NewInterpreter(m, o)
			v, err := VersionEdge()

			got := response{
				v:  v,
				er: err,
			}
			if got.er != tc.want.er {
				t.Errorf("Got %v but wants %v", got, tc.want.er)
			}
		})
	}
}

func testVerbosityEdge(t *testing.T) {
	type response struct {
		s Status
	}

	tests := []struct {
		name      string
		input     string
		imagePath string
		want      response
	}{
		{
			name:      "verbosity edge version",
			input:     "testing/mobilenet_v2_1.0_224_quant.tflite",
			imagePath: "testing/cat.png",
			want: response{
				s: StatusOk,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			m, _ := NewModelFromFile(tc.input)

			_ = NewEdge(Device{
				Type: TypeApexUSB,
				Path: "",
			})

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)
			_, _ = NewInterpreter(m, o)
			got := VerbosityEdge(0)

			if got != tc.want.s {
				t.Errorf("Got %v but wants %v", got, tc.want.s)
			}
		})
	}
}

func testDeviceListEdge(t *testing.T) {
	type response struct {
		d []Device
	}

	tests := []struct {
		name      string
		input     string
		imagePath string
		want      response
	}{
		{
			name:      "device list",
			input:     "testing/mobilenet_v2_1.0_224_quant.tflite",
			imagePath: "testing/cat.png",
			want: response{
				d: nil,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			m, _ := NewModelFromFile(tc.input)

			_ = NewEdge(Device{
				Type: TypeApexUSB,
				Path: "",
			})

			o, _ := NewInterpreterOptions()
			o.SetNumThread(1)
			_, _ = NewInterpreter(m, o)
			got := DeviceListEdge()

			if len(got) != len(tc.want.d) {
				t.Errorf("Got %v but wants %v", got, tc.want.d)
			}
		})
	}
}
