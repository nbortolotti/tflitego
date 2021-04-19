package tflite

import "testing"

func TestTFVersion(t *testing.T) {
	type response struct {
		v bool
	}

	tests := []struct {
		name string
		want response
	}{
		{
			name: "Success TFLite version",
			want: response{
				v: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got bool
			if Version() != "" {
				got = true
			}
			if got != tc.want.v {
				t.Errorf("expected: %v, got: %v", tc.want.v, got)
			}
		})
	}
}
