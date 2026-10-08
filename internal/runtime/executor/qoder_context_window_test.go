package executor

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

func TestConfiguredQoderContextLength(t *testing.T) {
	for _, model := range []string{"gmodel", "gfmodel", "smodel", "ultimate"} {
		for _, tc := range []struct {
			name    string
			raw     string
			wantErr bool
		}{
			{"published 1M", `{"context_config":{"200K":{"is_default":true,"token_count":200000},"1M":{"token_count":1000000}}}`, false},
			{"missing tiers", `{"max_input_tokens":1000000}`, true},
			{"1M no longer published", `{"max_input_tokens":1000000,"context_config":{"200K":{"token_count":200000}}}`, true},
			{"malformed tier", `{"context_config":{"1M":{"token_count":"1000000"}}}`, true},
			{"empty tiers", `{"context_config":{}}`, true},
			{"non-object tier", `{"context_config":{"1M":1000000}}`, true},
		} {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				var modelConfig map[string]interface{}
				if err := json.Unmarshal([]byte(tc.raw), &modelConfig); err != nil {
					t.Fatal(err)
				}
				got, err := configuredQoderContextLength(model, modelConfig)
				if tc.wantErr {
					if err == nil || got != 0 || !strings.Contains(err.Error(), "does not publish configured context_length 1000000") {
						t.Fatalf("configuredQoderContextLength() = %d, %v, want explicit catalog error", got, err)
					}
					return
				}
				if err != nil || got != 1000000 {
					t.Fatalf("configuredQoderContextLength() = %d, %v, want 1000000", got, err)
				}
			})
		}
	}
	for _, model := range []string{"auto", "efficient", "dfmodel", "qfmodel", "mode-custom"} {
		if got, err := configuredQoderContextLength(model, nil); err != nil || got != 0 {
			t.Fatalf("unselected model %s: got %d, %v, want unchanged selection", model, got, err)
		}
	}
}

func TestQoderParameters_ContextLength(t *testing.T) {
	params := qoderParameters(4096, "high", 1000000)
	if params["context_length"] != 1000000 || params["max_tokens"] != 4096 || params["reasoning_effort"] != "high" {
		t.Fatalf("parameters = %v, want context_length, output budget, and effort together", params)
	}
	if _, present := qoderParameters(4096, "", 0)["context_length"]; present {
		t.Fatal("unselected model must retain its previous wire shape")
	}
}

// Inspect the signed/encoded wire body, not just the parameters helper. A
// client does not need to know Qoder's proprietary context selection field.
func TestExecuteStream_QoderContextLengthOnWire(t *testing.T) {
	for _, model := range []string{"gmodel", "gfmodel", "smodel", "ultimate", "dfmodel"} {
		for _, source := range []sdktranslator.Format{sdktranslator.FormatOpenAI, sdktranslator.FormatClaude} {
			for _, clientWindow := range []int{0, 200000} {
				name := fmt.Sprintf("%s/%s/client-window-%d", model, source, clientWindow)
				t.Run(name, func(t *testing.T) {
					storage := testQoderStorageWithModelConfig()
					storage.SetModelConfigs(map[string]json.RawMessage{
						model: json.RawMessage(fmt.Sprintf(`{"key":%q,"source":"system","max_output_tokens":8192,"context_config":{"200K":{"is_default":true,"token_count":200000},"1M":{"token_count":1000000}},"thinking_config":{"enabled":{"efforts":{"high":{}}}}}`, model)),
					})
					var outgoing map[string]interface{}
					ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", qoderRoundTripFunc(func(req *http.Request) (*http.Response, error) {
						encoded, err := io.ReadAll(req.Body)
						if err != nil {
							return nil, err
						}
						decoded, err := decodeQoderContextTestBody(string(encoded))
						if err != nil {
							return nil, err
						}
						if err := json.Unmarshal(decoded, &outgoing); err != nil {
							return nil, err
						}
						return qoderSSEHTTPResponse(req, http.StatusOK, "data: [DONE]\n\n"), nil
					}))
					payload := map[string]interface{}{
						"model": "qoder/" + model, "messages": []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
						"max_tokens": 1024, "reasoning_effort": "high",
					}
					if source == sdktranslator.FormatClaude {
						payload["thinking"] = map[string]interface{}{"type": "enabled", "budget_tokens": 1024}
					}
					if clientWindow > 0 {
						payload["context_length"] = clientWindow
						payload["parameters"] = map[string]interface{}{"context_length": clientWindow}
					}
					body, _ := json.Marshal(payload)
					result, err := NewQoderExecutor(&config.Config{}).ExecuteStream(ctx,
						&cliproxyauth.Auth{Provider: "qoder", Storage: storage},
						cliproxyexecutor.Request{Model: "qoder/" + model, Payload: body},
						cliproxyexecutor.Options{SourceFormat: source})
					if err != nil {
						t.Fatal(err)
					}
					for range result.Chunks {
					}
					params, _ := outgoing["parameters"].(map[string]interface{})
					if model == "dfmodel" {
						if _, present := params["context_length"]; present {
							t.Fatal("unselected model's window selection changed")
						}
					} else if params["context_length"] != float64(1000000) {
						t.Fatalf("wire parameters.context_length = %v, want 1000000", params["context_length"])
					}
					if _, present := outgoing["context_length"]; present {
						t.Fatal("context_length must be nested in parameters")
					}
					if params["max_tokens"] != float64(1024) {
						t.Fatalf("output budget = %v, want 1024", params["max_tokens"])
					}
					if source == sdktranslator.FormatOpenAI && params["reasoning_effort"] != "high" {
						t.Fatalf("effort = %v, want high", params["reasoning_effort"])
					}
				})
			}
		}
	}
}

func TestExecuteStream_QoderContextLengthRejectsUnsupportedTier(t *testing.T) {
	storage := testQoderStorageWithModelConfig()
	storage.SetModelConfigs(map[string]json.RawMessage{
		"smodel": json.RawMessage(`{"key":"smodel","context_config":{"200K":{"token_count":200000}}}`),
	})
	calls := 0
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", qoderRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		return qoderSSEHTTPResponse(req, http.StatusOK, "data: [DONE]\n\n"), nil
	}))
	_, err := NewQoderExecutor(&config.Config{}).ExecuteStream(ctx,
		&cliproxyauth.Auth{Provider: "qoder", Storage: storage},
		cliproxyexecutor.Request{Model: "qoder/smodel", Payload: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
		cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI})
	if err == nil || !strings.Contains(err.Error(), "does not publish configured context_length 1000000") {
		t.Fatalf("error = %v, want explicit catalog error", err)
	}
	if calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func decodeQoderContextTestBody(encoded string) ([]byte, error) {
	const custom = "_doRTgHZBKcGVjlvpC,@aFSx#DPuNJme&i*MzLOEn)sUrthbf%Y^w.(kIQyXqWA!$"
	const standard = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
	translated := []byte(encoded)
	for i, c := range translated {
		index := strings.IndexByte(custom, c)
		if index < 0 {
			return nil, fmt.Errorf("unknown encoded character")
		}
		translated[i] = standard[index]
	}
	n := len(translated)
	a := n / 3
	restored := append([]byte{}, translated[n-a:]...)
	restored = append(restored, translated[a:n-a]...)
	restored = append(restored, translated[:a]...)
	return base64.StdEncoding.DecodeString(string(restored))
}
