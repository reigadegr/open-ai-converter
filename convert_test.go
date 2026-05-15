package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestConvertChatToResponses_MultiTurnWithToolCalls(t *testing.T) {
	chatReq := &ChatCompletionsRequest{
		Model: "gpt-4o",
		Messages: []ChatMessage{
			{Role: "system", Content: json.RawMessage(`"You are a helper."`)},
			{Role: "user", Content: json.RawMessage(`"What is the weather?"`)},
			{
				Role:    "assistant",
				Content: json.RawMessage(`null`),
				ToolCalls: []ToolCall{
					{
						ID:   "call_abc123",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"city":"Beijing"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				ToolCallID: "call_abc123",
				Content:    json.RawMessage(`"Sunny, 25°C"`),
			},
			{Role: "user", Content: json.RawMessage(`"Thanks, what about Shanghai?"`)},
		},
	}

	respReq, err := ConvertChatToResponsesRequest(chatReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if respReq.Model != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", respReq.Model)
	}

	// Instructions should come from system message
	if respReq.Instructions == nil || *respReq.Instructions != "You are a helper." {
		t.Errorf("instructions = %v", respReq.Instructions)
	}

	// Parse input (system messages go to Instructions, not input array)
	var inputs []ResponsesInputMessage
	json.Unmarshal(respReq.Input, &inputs)

	// First: user message
	if inputs[0].Role != "user" {
		t.Errorf("input[0] role = %v, want user", inputs[0].Role)
	}

	// Second: assistant function_call
	if inputs[1].Type != "function_call" {
		t.Errorf("input[1] type = %v, want function_call", inputs[1].Type)
	}
	if inputs[1].CallID != "call_abc123" {
		t.Errorf("input[1] call_id = %v, want call_abc123", inputs[1].CallID)
	}

	// Third: function_call_output
	if inputs[2].Type != "function_call_output" {
		t.Errorf("input[2] type = %v, want function_call_output", inputs[2].Type)
	}
	if inputs[2].CallID != "call_abc123" {
		t.Errorf("input[2] call_id = %v, want call_abc123", inputs[2].CallID)
	}

	// Fourth: user message
	if inputs[3].Role != "user" {
		t.Errorf("input[3] role = %v, want user", inputs[3].Role)
	}
}

func TestConvertChatToResponses_AssistantMessageHasID(t *testing.T) {
	chatReq := &ChatCompletionsRequest{
		Model: "gpt-4o",
		Messages: []ChatMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
			{Role: "assistant", Content: json.RawMessage(`"Hi there!"`)},
			{Role: "user", Content: json.RawMessage(`"How are you?"`)},
		},
	}

	respReq, err := ConvertChatToResponsesRequest(chatReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	var inputs []ResponsesInputMessage
	json.Unmarshal(respReq.Input, &inputs)

	// Find the assistant message (skip system instruction if any)
	for _, inp := range inputs {
		if inp.Role == "assistant" {
			// Assistant message content should be output_text
			var parts []map[string]interface{}
			json.Unmarshal(inp.Content, &parts)
			if len(parts) == 0 {
				t.Fatal("assistant content is empty")
			}
			if parts[0]["type"] != "output_text" {
				t.Errorf("assistant content type = %v, want output_text", parts[0]["type"])
			}
			return
		}
	}
	t.Error("no assistant message found in input")
}

func TestConvertResponsesToChat_MultiTurnWithToolCalls(t *testing.T) {
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"What is the weather?"}]},
		{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"get_weather","arguments":"{\"city\":\"Beijing\"}","status":"completed"},
		{"type":"function_call_output","call_id":"call_abc","output":"Sunny, 25°C"},
		{"type":"message","id":"msg_2","role":"user","status":"completed","content":[{"type":"input_text","text":"Thanks!"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if chatReq.Model != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", chatReq.Model)
	}

	// First: user message
	if chatReq.Messages[0].Role != "user" {
		t.Errorf("msg[0] role = %v, want user", chatReq.Messages[0].Role)
	}

	// Second: assistant with tool_calls
	if chatReq.Messages[1].Role != "assistant" {
		t.Errorf("msg[1] role = %v, want assistant", chatReq.Messages[1].Role)
	}
	if len(chatReq.Messages[1].ToolCalls) != 1 {
		t.Errorf("msg[1] tool_calls count = %d, want 1", len(chatReq.Messages[1].ToolCalls))
	}

	// Third: tool response
	if chatReq.Messages[2].Role != "tool" {
		t.Errorf("msg[2] role = %v, want tool", chatReq.Messages[2].Role)
	}
	if chatReq.Messages[2].ToolCallID != "call_abc" {
		t.Errorf("msg[2] tool_call_id = %v, want call_abc", chatReq.Messages[2].ToolCallID)
	}

	// Fourth: user message
	if chatReq.Messages[3].Role != "user" {
		t.Errorf("msg[3] role = %v, want user", chatReq.Messages[3].Role)
	}
}

func TestConvertResponsesToChat_AssistantMessageOutputText(t *testing.T) {
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"message","id":"msg_2","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hi there!"}]},
		{"type":"message","id":"msg_3","role":"user","status":"completed","content":[{"type":"input_text","text":"How are you?"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Assistant message content
	assistantMsg := chatReq.Messages[1]
	if assistantMsg.Role != "assistant" {
		t.Errorf("msg[1] role = %v, want assistant", assistantMsg.Role)
	}

	var content []map[string]interface{}
	json.Unmarshal(assistantMsg.Content, &content)
	if len(content) != 1 {
		t.Fatalf("msg[1] content count = %d, want 1", len(content))
	}
	if content[0]["type"] != "text" {
		t.Errorf("content type = %v, want text", content[0]["type"])
	}
	if content[0]["text"] != "Hi there!" {
		t.Errorf("content text = %v, want 'Hi there!'", content[0]["text"])
	}
}

func TestConvertResponsesToChat_MissingRoleDefaultsToAssistant(t *testing.T) {
	inputJSON := `[
		{"type":"message","id":"msg_1","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"message","id":"msg_2","content":[{"type":"output_text","text":"Hi!"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if chatReq.Messages[0].Role != "assistant" {
		t.Errorf("msg[0] role = %v, want assistant (default)", chatReq.Messages[0].Role)
	}
}

func TestConvertChatToResponses_MultipleSystemMessages(t *testing.T) {
	chatReq := &ChatCompletionsRequest{
		Model: "gpt-4o",
		Messages: []ChatMessage{
			{Role: "system", Content: json.RawMessage(`"System prompt."`)},
			{Role: "system", Content: json.RawMessage(`"Additional instructions."`)},
			{Role: "developer", Content: json.RawMessage(`"Developer note."`)},
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
	}

	respReq, err := ConvertChatToResponsesRequest(chatReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if respReq.Instructions == nil {
		t.Fatal("instructions is nil")
	}
	instructions := *respReq.Instructions
	if !strings.Contains(instructions, "System prompt.") {
		t.Errorf("instructions missing 'System prompt.'")
	}
	if !strings.Contains(instructions, "Additional instructions.") {
		t.Errorf("instructions missing 'Additional instructions.'")
	}
	if !strings.Contains(instructions, "Developer note.") {
		t.Errorf("instructions missing 'Developer note.'")
	}
	if !strings.Contains(instructions, "\n\n") {
		t.Errorf("instructions should be joined with \\n\\n")
	}
}

func TestConvertResponsesToChat_RefusalContent(t *testing.T) {
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Do something bad"}]},
		{"type":"message","id":"msg_2","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"I cannot help with that."}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	msg := chatReq.Messages[1]
	if msg.Refusal == nil || *msg.Refusal != "I cannot help with that." {
		t.Errorf("refusal = %v, want 'I cannot help with that.'", msg.Refusal)
	}
}

func TestConvertResponsesToChat_NoStatusField(t *testing.T) {
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Serialize to JSON and check no "status" field
	b, _ := json.Marshal(chatReq.Messages[0])
	var m map[string]interface{}
	json.Unmarshal(b, &m)
	if _, exists := m["status"]; exists {
		t.Errorf("msg should not have 'status' field")
	}
}

func TestConvertChatRespToResponsesResp_ToolCallOnlyNoEmptyMessage(t *testing.T) {
	chatResp := &ChatCompletionsResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1000,
		Model:   "gpt-4o",
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: &ChatMessage{
					Role:    "assistant",
					Content: json.RawMessage(`null`),
					ToolCalls: []ToolCall{
						{
							ID:   "call_xyz",
							Type: "function",
							Function: FunctionCall{
								Name:      "get_weather",
								Arguments: `{"city":"Beijing"}`,
							},
						},
					},
				},
				FinishReason: strPtr("tool_calls"),
			},
		},
	}

	respResp, err := ConvertChatRespToResponsesResp(chatResp)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	for _, item := range respResp.Output {
		if item.Type == "message" {
			t.Errorf("should not have message output item for tool-call-only response")
		}
	}

	if len(respResp.Output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(respResp.Output))
	}
	if respResp.Output[0].Type != "function_call" {
		t.Errorf("output[0] type = %v, want function_call", respResp.Output[0].Type)
	}
}

func TestConvertChatContentToResponses_ImageURL(t *testing.T) {
	content := json.RawMessage(`[{"type":"text","text":"Describe this"},{"type":"image_url","image_url":{"url":"https://example.com/cat.jpg","detail":"high"}}]`)

	result := convertChatContentToResponses(content)
	parts, ok := result.([]map[string]interface{})
	if !ok {
		t.Fatalf("expected []map[string]interface{}, got %T", result)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}

	if parts[0]["type"] != "input_text" {
		t.Errorf("parts[0] type = %v, want input_text", parts[0]["type"])
	}
	if parts[1]["type"] != "input_image" {
		t.Errorf("parts[1] type = %v, want input_image", parts[1]["type"])
	}
}

func TestGenerateID_Uniqueness(t *testing.T) {
	seen := make(map[string]bool)
	for i := 0; i < 1000; i++ {
		id := generateID("test_")
		if seen[id] {
			t.Fatalf("duplicate ID generated: %s", id)
		}
		seen[id] = true
	}
}

func TestConvertChatToResponses_StopParameter(t *testing.T) {
	stop := json.RawMessage(`["\n","STOP"]`)
	chatReq := &ChatCompletionsRequest{
		Model:  "gpt-4o",
		Stop:   stop,
		Stream: false,
		Messages: []ChatMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
	}

	respReq, err := ConvertChatToResponsesRequest(chatReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if respReq.Stop == nil {
		t.Errorf("stop parameter should be passed through")
	}
}

func TestConvertChatToResponses_ToolStrictInParameters(t *testing.T) {
	strict := true
	chatReq := &ChatCompletionsRequest{
		Model: "gpt-4o",
		Messages: []ChatMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Tools: []ChatTool{
			{
				Type: "function",
				Function: ChatFunction{
					Name:        "test_func",
					Description: "A test function",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"arg":{"type":"string"}},"required":["arg"]}`),
					Strict:      &strict,
				},
			},
		},
	}

	respReq, err := ConvertChatToResponsesRequest(chatReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Parse tools
	var tools []ResponsesTool
	json.Unmarshal(respReq.Tools, &tools)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	if tools[0].Strict == nil || *tools[0].Strict != true {
		t.Errorf("strict should be at tool level and be true")
	}
}

func TestConvertChatToResponses_EmptyResponseFormatIgnored(t *testing.T) {
	chatReq := &ChatCompletionsRequest{
		Model: "gpt-4o",
		Messages: []ChatMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		ResponseFormat: json.RawMessage(`{"type":""}`),
	}

	respReq, err := ConvertChatToResponsesRequest(chatReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if respReq.Text != nil {
		t.Errorf("text should not be set for empty type response_format")
	}
}

func TestConvertResponsesToChat_TextWithOnlyVerbosity(t *testing.T) {
	textJSON := `{"verbosity": "high"}`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]}]`),
		Text:  json.RawMessage(textJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if chatReq.ResponseFormat != nil {
		t.Errorf("response_format should not be set for text with only verbosity")
	}
}

func TestConvertResponsesToChat_ToolStrictFromToolLevel(t *testing.T) {
	toolsJSON := `[
		{
			"type": "function",
			"name": "test_func",
			"description": "A test function",
			"strict": false,
			"parameters": {
				"type": "object",
				"properties": {"arg": {"type": "string"}},
				"required": ["arg"]
			}
		}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]}]`),
		Tools: json.RawMessage(toolsJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if len(chatReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(chatReq.Tools))
	}

	if chatReq.Tools[0].Function.Strict == nil || *chatReq.Tools[0].Function.Strict != false {
		t.Errorf("function.strict = %v, want false", chatReq.Tools[0].Function.Strict)
	}
}

func TestConvertResponsesToChat_ToolStrictFromInsideParameters(t *testing.T) {
	toolsJSON := `[
		{
			"type": "function",
			"name": "test_func",
			"parameters": {
				"type": "object",
				"properties": {"arg": {"type": "string"}},
				"strict": true
			}
		}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]}]`),
		Tools: json.RawMessage(toolsJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if len(chatReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(chatReq.Tools))
	}

	fn := chatReq.Tools[0].Function
	if fn.Strict == nil || *fn.Strict != true {
		t.Errorf("function.strict = %v, want true", fn.Strict)
	}
}

func TestConvertResponsesToChat_NamespaceToolsFlattened(t *testing.T) {
	toolsJSON := `[
		{
			"type": "function",
			"name": "regular_tool",
			"description": "A regular tool",
			"parameters": {"type": "object", "properties": {}}
		},
		{
			"type": "namespace",
			"name": "mcp__context7__",
			"description": "Context7 MCP server",
			"tools": [
				{
					"type": "function",
					"name": "query_docs",
					"description": "Query documentation",
					"strict": false,
					"parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
				},
				{
					"type": "function",
					"name": "resolve_id",
					"description": "Resolve library ID",
					"strict": false,
					"parameters": {"type": "object", "properties": {"name": {"type": "string"}}}
				}
			]
		}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]}]`),
		Tools: json.RawMessage(toolsJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if len(chatReq.Tools) != 3 {
		t.Fatalf("expected 3 tools (1 regular + 2 namespace), got %d", len(chatReq.Tools))
	}

	if chatReq.Tools[0].Function.Name != "regular_tool" {
		t.Errorf("tool[0] name = %v, want regular_tool", chatReq.Tools[0].Function.Name)
	}
	if chatReq.Tools[1].Function.Name != "mcp__context7__query_docs" {
		t.Errorf("tool[1] name = %v, want mcp__context7__query_docs", chatReq.Tools[1].Function.Name)
	}
	if chatReq.Tools[2].Function.Name != "mcp__context7__resolve_id" {
		t.Errorf("tool[2] name = %v, want mcp__context7__resolve_id", chatReq.Tools[2].Function.Name)
	}
}

// ==================== Stream Options Tests ====================

func TestConvertChatToResponses_StreamOptions(t *testing.T) {
	includeUsage := true
	req := &ChatCompletionsRequest{
		Model:  "gpt-4",
		Stream: true,
		StreamOptions: &StreamOptions{
			IncludeUsage: includeUsage,
		},
		Messages: []ChatMessage{
			{Role: "user", Content: jsonString("hello")},
		},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	if !respReq.Stream {
		t.Errorf("stream = %v, want true", respReq.Stream)
	}
	if respReq.StreamOptions == nil || !respReq.StreamOptions.IncludeUsage {
		t.Errorf("stream_options.include_usage should be true")
	}
}

func TestConvertChatToResponses_StreamOptionsAutoInclude(t *testing.T) {
	req := &ChatCompletionsRequest{
		Model:  "gpt-4",
		Stream: true,
		Messages: []ChatMessage{
			{Role: "user", Content: jsonString("hello")},
		},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	if !respReq.Stream {
		t.Errorf("stream = %v, want true", respReq.Stream)
	}
	if respReq.StreamOptions == nil || !respReq.StreamOptions.IncludeUsage {
		t.Errorf("stream_options.include_usage should be auto-added")
	}
}

func TestConvertChatToResponses_NoStreamOptionsWhenNotStreaming(t *testing.T) {
	req := &ChatCompletionsRequest{
		Model:  "gpt-4",
		Stream: false,
		Messages: []ChatMessage{
			{Role: "user", Content: jsonString("hello")},
		},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	if respReq.StreamOptions != nil {
		t.Error("stream_options should not be present when stream=false")
	}
}

func TestConvertResponsesToChat_StreamOptions(t *testing.T) {
	req := &ResponsesRequest{
		Model:  "gpt-4",
		Stream: true,
		Input:  jsonString("hello"),
	}
	chatReq, err := ConvertResponsesToChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	if !chatReq.Stream {
		t.Errorf("stream = %v, want true", chatReq.Stream)
	}
	if chatReq.StreamOptions == nil || !chatReq.StreamOptions.IncludeUsage {
		t.Errorf("stream_options.include_usage should be true")
	}
}

// ==================== New Tests ====================

func TestConvertChatToResponses_MaxTokensClamped(t *testing.T) {
	maxTokens := 50
	req := &ChatCompletionsRequest{
		Model:      "gpt-4",
		MaxTokens:  &maxTokens,
		Messages:   []ChatMessage{{Role: "user", Content: jsonString("hello")}},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	if respReq.MaxOutputTokens == nil || *respReq.MaxOutputTokens != minMaxOutputTokens {
		t.Errorf("max_output_tokens = %v, want %d (clamped)", respReq.MaxOutputTokens, minMaxOutputTokens)
	}
}

func TestConvertChatToResponses_ReasoningEffort(t *testing.T) {
	effort := "high"
	req := &ChatCompletionsRequest{
		Model:          "o3",
		ReasoningEffort: &effort,
		Messages:       []ChatMessage{{Role: "user", Content: jsonString("hello")}},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	if respReq.Reasoning == nil {
		t.Fatal("reasoning should be set")
	}
	if respReq.Reasoning.Effort != "high" {
		t.Errorf("reasoning.effort = %v, want high", respReq.Reasoning.Effort)
	}
	if respReq.Reasoning.Summary != "auto" {
		t.Errorf("reasoning.summary = %v, want auto", respReq.Reasoning.Summary)
	}
}

func TestConvertChatToResponses_EmptyBase64ImageSkipped(t *testing.T) {
	content := json.RawMessage(`[{"type":"text","text":"test"},{"type":"image_url","image_url":{"url":"data:image/png;base64,   ","detail":"auto"}}]`)
	req := &ChatCompletionsRequest{
		Model:    "gpt-4o",
		Messages: []ChatMessage{{Role: "user", Content: content}},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	var inputs []ResponsesInputMessage
	json.Unmarshal(respReq.Input, &inputs)

	// Find user message
	for _, inp := range inputs {
		if inp.Role == "user" {
			var parts []map[string]interface{}
			json.Unmarshal(inp.Content, &parts)
			if len(parts) != 1 {
				t.Errorf("expected 1 part (empty image skipped), got %d", len(parts))
			}
			if len(parts) > 0 && parts[0]["type"] != "input_text" {
				t.Errorf("expected input_text, got %v", parts[0]["type"])
			}
			return
		}
	}
	t.Error("no user message found")
}

func TestConvertChatToResponses_ThinkingContentWrapped(t *testing.T) {
	content := json.RawMessage(`[{"type":"text","text":"Hello"},{"type":"thinking","thinking":"Let me think..."}]`)
	req := &ChatCompletionsRequest{
		Model:    "gpt-4o",
		Messages: []ChatMessage{{Role: "assistant", Content: content}},
	}
	respReq, err := ConvertChatToResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}

	var inputs []ResponsesInputMessage
	json.Unmarshal(respReq.Input, &inputs)

	for _, inp := range inputs {
		if inp.Role == "assistant" {
			var parts []map[string]interface{}
			json.Unmarshal(inp.Content, &parts)
			if len(parts) == 0 {
				t.Fatal("assistant content is empty")
			}
			text, _ := parts[0]["text"].(string)
			if !strings.Contains(text, "<thinking>") || !strings.Contains(text, "</thinking>") {
				t.Errorf("thinking should be wrapped in tags, got: %s", text)
			}
			if !strings.Contains(text, "Hello") {
				t.Errorf("text content should be preserved")
			}
			return
		}
	}
	t.Error("no assistant message found")
}

func TestConvertResponsesRespToChatResp_ReasoningOutput(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_123",
		Status: "completed",
		Model:  "o3",
		Output: []OutputItem{
			{
				Type: "reasoning",
				Summary: []ResponsesSummary{
					{Type: "summary_text", Text: "I thought about this."},
				},
			},
			{
				Type: "message",
				Role: "assistant",
				Content: []ContentPart{
					{Type: "output_text", Text: "The answer is 42."},
				},
			},
		},
	}

	chatResp, err := ConvertResponsesRespToChatResp(resp)
	if err != nil {
		t.Fatal(err)
	}

	msg := chatResp.Choices[0].Message
	if msg.ReasoningContent == nil || *msg.ReasoningContent != "I thought about this." {
		t.Errorf("reasoning_content = %v, want 'I thought about this.'", msg.ReasoningContent)
	}

	text := contentToString(msg.Content)
	if text != "The answer is 42." {
		t.Errorf("content = %v, want 'The answer is 42.'", text)
	}
}

func TestConvertResponsesRespToChatResp_WebSearchCallFiltered(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_123",
		Status: "completed",
		Output: []OutputItem{
			{
				Type:  "web_search_call",
				ID:    "ws_1",
				Action: &WebSearchAction{Type: "search", Query: "test"},
			},
			{
				Type: "message",
				Role: "assistant",
				Content: []ContentPart{
					{Type: "output_text", Text: "Search result text."},
				},
			},
		},
	}

	chatResp, err := ConvertResponsesRespToChatResp(resp)
	if err != nil {
		t.Fatal(err)
	}

	text := contentToString(chatResp.Choices[0].Message.Content)
	if text != "Search result text." {
		t.Errorf("content = %v, want 'Search result text.'", text)
	}
}

func TestConvertResponsesRespToChatResp_IncompleteDetails(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_123",
		Status: "incomplete",
		Model:  "gpt-4o",
		Output: []OutputItem{
			{
				Type: "message",
				Role: "assistant",
				Content: []ContentPart{
					{Type: "output_text", Text: "Partial..."},
				},
			},
		},
		IncompleteDetails: &ResponsesIncompleteDetails{Reason: "max_output_tokens"},
	}

	chatResp, err := ConvertResponsesRespToChatResp(resp)
	if err != nil {
		t.Fatal(err)
	}

	if *chatResp.Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason = %v, want length", *chatResp.Choices[0].FinishReason)
	}
}

// ==================== Streaming State Machine Tests ====================

func TestResponsesEventToChatChunks_TextDelta(t *testing.T) {
	state := NewResponsesEventToChatState()
	state.Model = "gpt-4o"

	// First: response.created
	evt := &ResponsesStreamEvent{
		Type: "response.created",
		Response: &ResponsesResponse{
			ID:    "resp_123",
			Model: "gpt-4o",
		},
	}
	chunks := ResponsesEventToChatChunks(evt, state)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk for created, got %d", len(chunks))
	}
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("role = %v, want assistant", chunks[0].Choices[0].Delta.Role)
	}

	// Text delta
	evt = &ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Hello",
	}
	chunks = ResponsesEventToChatChunks(evt, state)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Choices[0].Delta.Content == nil || *chunks[0].Choices[0].Delta.Content != "Hello" {
		t.Errorf("content = %v, want Hello", chunks[0].Choices[0].Delta.Content)
	}
}

func TestResponsesEventToChatChunks_ToolCallDelta(t *testing.T) {
	state := NewResponsesEventToChatState()
	state.Model = "gpt-4o"

	// Output item added (function_call)
	evt := &ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &OutputItem{
			Type:   "function_call",
			CallID: "call_123",
			Name:   "get_weather",
		},
	}
	chunks := ResponsesEventToChatChunks(evt, state)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	tc := chunks[0].Choices[0].Delta.ToolCalls[0]
	if tc.ID != "call_123" {
		t.Errorf("tool call ID = %v, want call_123", tc.ID)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("function name = %v, want get_weather", tc.Function.Name)
	}

	// Args delta
	evt = &ResponsesStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"city":"`,
	}
	chunks = ResponsesEventToChatChunks(evt, state)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments != `{"city":"` {
		t.Errorf("arguments = %v", chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments)
	}
}

func TestResponsesEventToChatChunks_Completed(t *testing.T) {
	state := NewResponsesEventToChatState()
	state.Model = "gpt-4o"
	state.SawText = true

	evt := &ResponsesStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResponse{
			Status: "completed",
			Usage: &ResponsesUsage{
				InputTokens:  100,
				OutputTokens: 50,
				TotalTokens:  150,
			},
		},
	}
	chunks := ResponsesEventToChatChunks(evt, state)
	if len(chunks) < 1 {
		t.Fatal("expected at least 1 chunk")
	}

	// First chunk: finish
	if chunks[0].Choices[0].FinishReason == nil || *chunks[0].Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", chunks[0].Choices[0].FinishReason)
	}

	// Second chunk: usage
	if len(chunks) >= 2 && chunks[1].Usage != nil {
		if chunks[1].Usage.PromptTokens != 100 {
			t.Errorf("prompt_tokens = %v, want 100", chunks[1].Usage.PromptTokens)
		}
	}
}

func TestFinalizeResponsesChatStream(t *testing.T) {
	state := NewResponsesEventToChatState()
	state.Model = "gpt-4o"
	state.SawToolCall = true

	chunks := FinalizeResponsesChatStream(state)
	if len(chunks) == 0 {
		t.Fatal("expected chunks from finalize")
	}

	if chunks[0].Choices[0].FinishReason == nil || *chunks[0].Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", chunks[0].Choices[0].FinishReason)
	}

	// Idempotent: second call returns nil
	chunks2 := FinalizeResponsesChatStream(state)
	if chunks2 != nil {
		t.Errorf("second finalize should return nil, got %d chunks", len(chunks2))
	}
}

// ==================== Reasoning Input Tests ====================

func TestConvertResponsesToChat_ReasoningInput(t *testing.T) {
	inputJSON := `[
		{"type":"reasoning","id":"r_1","summary":[{"type":"summary_text","text":"I considered the problem carefully."}]},
		{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"The answer is 42."}]}
	]`

	respReq := &ResponsesRequest{
		Model: "o3",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Should have one assistant message
	if len(chatReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(chatReq.Messages))
	}

	msg := chatReq.Messages[0]
	if msg.Role != "assistant" {
		t.Errorf("role = %v, want assistant", msg.Role)
	}

	// ReasoningContent should contain the summary text
	if msg.ReasoningContent == nil {
		t.Fatal("reasoning_content is nil, want 'I considered the problem carefully.'")
	}
	if *msg.ReasoningContent != "I considered the problem carefully." {
		t.Errorf("reasoning_content = %v, want 'I considered the problem carefully.'", *msg.ReasoningContent)
	}

	// Content should be the output_text
	text := contentToString(msg.Content)
	if text != "The answer is 42." {
		t.Errorf("content = %v, want 'The answer is 42.'", text)
	}
}

func TestConvertResponsesToChat_ReasoningThenToolCall(t *testing.T) {
	inputJSON := `[
		{"type":"reasoning","id":"r_1","summary":[{"type":"summary_text","text":"I need to check the weather."}]},
		{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"get_weather","arguments":"{\"city\":\"Beijing\"}","status":"completed"},
		{"type":"function_call_output","call_id":"call_abc","output":"Sunny, 25°C"}
	]`

	respReq := &ResponsesRequest{
		Model: "o3",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Should have 2 messages: assistant (with tool_calls + reasoning) + tool result
	if len(chatReq.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
	}

	// First: assistant with both reasoning and tool_calls
	assistantMsg := chatReq.Messages[0]
	if assistantMsg.Role != "assistant" {
		t.Errorf("msg[0] role = %v, want assistant", assistantMsg.Role)
	}

	if len(assistantMsg.ToolCalls) != 1 {
		t.Fatalf("msg[0] tool_calls count = %d, want 1", len(assistantMsg.ToolCalls))
	}
	if assistantMsg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool_call name = %v, want get_weather", assistantMsg.ToolCalls[0].Function.Name)
	}
	if assistantMsg.ToolCalls[0].ID != "call_abc" {
		t.Errorf("tool_call id = %v, want call_abc", assistantMsg.ToolCalls[0].ID)
	}

	if assistantMsg.ReasoningContent == nil {
		t.Fatal("msg[0] reasoning_content is nil, want 'I need to check the weather.'")
	}
	if *assistantMsg.ReasoningContent != "I need to check the weather." {
		t.Errorf("reasoning_content = %v, want 'I need to check the weather.'", *assistantMsg.ReasoningContent)
	}

	// Second: tool result
	toolMsg := chatReq.Messages[1]
	if toolMsg.Role != "tool" {
		t.Errorf("msg[1] role = %v, want tool", toolMsg.Role)
	}
	if toolMsg.ToolCallID != "call_abc" {
		t.Errorf("msg[1] tool_call_id = %v, want call_abc", toolMsg.ToolCallID)
	}
}

// ==================== cleanupOrphanedToolCalls Tests ====================

func TestCleanupOrphanedToolCalls(t *testing.T) {
	t.Run("matched tool result: keep both", func(t *testing.T) {
		msgs := []ChatMessage{
			{
				Role:    "assistant",
				Content: json.RawMessage("null"),
				ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: "f1", Arguments: "{}"}},
				},
			},
			{Role: "tool", ToolCallID: "call_1", Content: jsonString("result 1")},
		}
		result := cleanupOrphanedToolCalls(msgs)
		if len(result) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(result))
		}
		if len(result[0].ToolCalls) != 1 {
			t.Errorf("expected 1 tool_call, got %d", len(result[0].ToolCalls))
		}
	})

	t.Run("orphaned tool_call with content: keep message, remove tool_calls", func(t *testing.T) {
		msgs := []ChatMessage{
			{
				Role:    "assistant",
				Content: jsonString("I changed my mind."),
				ToolCalls: []ToolCall{
					{ID: "call_orphan", Type: "function", Function: FunctionCall{Name: "f1", Arguments: "{}"}},
				},
			},
		}
		result := cleanupOrphanedToolCalls(msgs)
		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		if len(result[0].ToolCalls) != 0 {
			t.Errorf("tool_calls should be removed, got %d", len(result[0].ToolCalls))
		}
		if contentToString(result[0].Content) != "I changed my mind." {
			t.Errorf("content should be preserved")
		}
	})

	t.Run("orphaned tool_call with reasoning_content: keep message, remove tool_calls", func(t *testing.T) {
		msgs := []ChatMessage{
			{
				Role:             "assistant",
				Content:          json.RawMessage("null"),
				ReasoningContent: strPtr("Let me think..."),
				ToolCalls: []ToolCall{
					{ID: "call_orphan", Type: "function", Function: FunctionCall{Name: "f1", Arguments: "{}"}},
				},
			},
		}
		result := cleanupOrphanedToolCalls(msgs)
		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		if len(result[0].ToolCalls) != 0 {
			t.Errorf("tool_calls should be removed, got %d", len(result[0].ToolCalls))
		}
		if result[0].ReasoningContent == nil || *result[0].ReasoningContent != "Let me think..." {
			t.Errorf("reasoning_content should be preserved")
		}
	})

	t.Run("orphaned tool_call with refusal: keep message, remove tool_calls", func(t *testing.T) {
		msgs := []ChatMessage{
			{
				Role:    "assistant",
				Content: json.RawMessage("null"),
				Refusal: strPtr("I cannot help with that request."),
				ToolCalls: []ToolCall{
					{ID: "call_orphan", Type: "function", Function: FunctionCall{Name: "f1", Arguments: "{}"}},
				},
			},
		}
		result := cleanupOrphanedToolCalls(msgs)
		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		if len(result[0].ToolCalls) != 0 {
			t.Errorf("tool_calls should be removed, got %d", len(result[0].ToolCalls))
		}
		if result[0].Refusal == nil || *result[0].Refusal != "I cannot help with that request." {
			t.Errorf("refusal should be preserved, got %v", result[0].Refusal)
		}
	})

	t.Run("orphaned tool_call with no content or reasoning: delete message", func(t *testing.T) {
		msgs := []ChatMessage{
			{
				Role:    "assistant",
				Content: json.RawMessage("null"),
				ToolCalls: []ToolCall{
					{ID: "call_orphan", Type: "function", Function: FunctionCall{Name: "f1", Arguments: "{}"}},
				},
			},
		}
		result := cleanupOrphanedToolCalls(msgs)
		if len(result) != 0 {
			t.Fatalf("expected 0 messages, got %d", len(result))
		}
	})

	t.Run("orphaned tool result: delete", func(t *testing.T) {
		msgs := []ChatMessage{
			{
				Role:    "assistant",
				Content: json.RawMessage("null"),
				ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: "f1", Arguments: "{}"}},
				},
			},
			{Role: "tool", ToolCallID: "call_1", Content: jsonString("result 1")},
			{Role: "tool", ToolCallID: "call_orphan", Content: jsonString("stale result")},
		}
		result := cleanupOrphanedToolCalls(msgs)
		if len(result) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(result))
		}
		// The orphaned tool result should be gone
		for _, m := range result {
			if m.Role == "tool" && m.ToolCallID == "call_orphan" {
				t.Error("orphaned tool result should have been removed")
			}
		}
	})
}

// ==================== Developer Role Test ====================

func TestConvertResponsesToChat_DeveloperRole(t *testing.T) {
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"developer","status":"completed","content":[{"type":"input_text","text":"Be concise."}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if len(chatReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(chatReq.Messages))
	}

	msg := chatReq.Messages[0]
	if msg.Role != "system" {
		t.Errorf("role = %v, want system (developer should be converted to system)", msg.Role)
	}

	text := contentToString(msg.Content)
	if text != "Be concise." {
		t.Errorf("content = %v, want 'Be concise.'", text)
	}
}

// ==================== ReasoningNil Response Test ====================

func TestConvertChatRespToResponsesResp_ReasoningNil(t *testing.T) {
	chatResp := &ChatCompletionsResponse{
		ID:      "chatcmpl-nil-reasoning",
		Object:  "chat.completion",
		Created: 1000,
		Model:   "gpt-4o",
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: &ChatMessage{
					Role:             "assistant",
					Content:          jsonString("Hello!"),
					ReasoningContent: nil,
				},
				FinishReason: strPtr("stop"),
			},
		},
	}

	respResp, err := ConvertChatRespToResponsesResp(chatResp)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Should not contain any reasoning output items
	for _, item := range respResp.Output {
		if item.Type == "reasoning" {
			t.Errorf("should not have reasoning output item when ReasoningContent is nil")
		}
	}

	// Should have exactly 1 message output item
	if len(respResp.Output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(respResp.Output))
	}
	if respResp.Output[0].Type != "message" {
		t.Errorf("output[0] type = %v, want message", respResp.Output[0].Type)
	}
}

// ==================== Empty Assistant Message Prevention Tests ====================

func TestConvertResponsesToChat_EmptyAssistantMessageSkipped(t *testing.T) {
	// An assistant message with no content should be skipped entirely
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"message","id":"msg_2","role":"assistant","status":"completed"},
		{"type":"message","id":"msg_3","role":"user","status":"completed","content":[{"type":"input_text","text":"How are you?"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// The empty assistant message should be dropped, leaving only the 2 user messages
	if len(chatReq.Messages) != 2 {
		t.Fatalf("expected 2 messages (empty assistant dropped), got %d", len(chatReq.Messages))
	}
	if chatReq.Messages[0].Role != "user" {
		t.Errorf("msg[0] role = %v, want user", chatReq.Messages[0].Role)
	}
	if chatReq.Messages[1].Role != "user" {
		t.Errorf("msg[1] role = %v, want user", chatReq.Messages[1].Role)
	}
}

func TestConvertResponsesToChat_EmptyContentAssistantSkipped(t *testing.T) {
	// An assistant message with empty content array should be skipped
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"message","id":"msg_2","role":"assistant","status":"completed","content":[]},
		{"type":"message","id":"msg_3","role":"user","status":"completed","content":[{"type":"input_text","text":"How are you?"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	if len(chatReq.Messages) != 2 {
		t.Fatalf("expected 2 messages (empty-content assistant dropped), got %d", len(chatReq.Messages))
	}
}

func TestConvertResponsesToChat_ReasoningOnlyProducesAssistantMessage(t *testing.T) {
	// A standalone reasoning item (not followed by assistant message or tool call)
	// should produce an assistant message with reasoning_content
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"reasoning","id":"r_1","summary":[{"type":"summary_text","text":"I considered the question carefully."}]},
		{"type":"message","id":"msg_2","role":"user","status":"completed","content":[{"type":"input_text","text":"Follow up"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "o3",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// Should have: user, assistant (reasoning only), user
	if len(chatReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
	}

	// The reasoning should appear as an assistant message between the two user messages
	assistantMsg := chatReq.Messages[1]
	if assistantMsg.Role != "assistant" {
		t.Errorf("msg[1] role = %v, want assistant", assistantMsg.Role)
	}
	if assistantMsg.ReasoningContent == nil || *assistantMsg.ReasoningContent != "I considered the question carefully." {
		t.Errorf("msg[1] reasoning_content = %v, want 'I considered the question carefully.'", assistantMsg.ReasoningContent)
	}
	// Verify no empty content is sent
	if assistantMsg.Content != nil {
		t.Errorf("msg[1] content should be nil for reasoning-only message, got %v", string(assistantMsg.Content))
	}
}

func TestConvertResponsesToChat_AllEmptyAssistantMessagesNeverProduced(t *testing.T) {
	// Comprehensive test: ensure no assistant message has all three fields empty
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"reasoning","id":"r_1","summary":[{"type":"summary_text","text":"Thinking..."}]},
		{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"get_weather","arguments":"{\"city\":\"Beijing\"}","status":"completed"},
		{"type":"function_call_output","call_id":"call_abc","output":"Sunny, 25°C"},
		{"type":"message","id":"msg_2","role":"assistant","status":"completed"},
		{"type":"message","id":"msg_3","role":"user","status":"completed","content":[{"type":"input_text","text":"Thanks"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "o3",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	for i, msg := range chatReq.Messages {
		if msg.Role == "assistant" {
			hasContent := msg.Content != nil && contentToString(msg.Content) != ""
			hasToolCalls := len(msg.ToolCalls) > 0
			hasReasoning := msg.ReasoningContent != nil && *msg.ReasoningContent != ""
			hasRefusal := msg.Refusal != nil && *msg.Refusal != ""
			if !hasContent && !hasToolCalls && !hasReasoning && !hasRefusal {
				t.Errorf("msg[%d] is an empty assistant message (no content, tool_calls, reasoning_content, or refusal)", i)
			}
		}
	}
}

func TestConvertResponsesToChat_OrphanedToolCallWithReasoningKept(t *testing.T) {
	// Reasoning + function_call without function_call_output:
	// tool_calls should be stripped but reasoning_content preserved
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"reasoning","id":"r_1","summary":[{"type":"summary_text","text":"I need to check something."}]},
		{"type":"function_call","id":"fc_1","call_id":"call_orphan","name":"get_weather","arguments":"{}","status":"completed"},
		{"type":"message","id":"msg_2","role":"user","status":"completed","content":[{"type":"input_text","text":"Follow up"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "o3",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// The assistant message should have reasoning_content but no orphaned tool_calls
	found := false
	for _, msg := range chatReq.Messages {
		if msg.Role == "assistant" {
			found = true
			if msg.ReasoningContent == nil || *msg.ReasoningContent != "I need to check something." {
				t.Errorf("reasoning_content should be preserved, got %v", msg.ReasoningContent)
			}
			if len(msg.ToolCalls) != 0 {
				t.Errorf("orphaned tool_calls should be removed, got %d", len(msg.ToolCalls))
			}
			hasContent := msg.Content != nil && contentToString(msg.Content) != ""
			hasReasoning := msg.ReasoningContent != nil && *msg.ReasoningContent != ""
			hasRefusal := msg.Refusal != nil && *msg.Refusal != ""
			if !hasContent && !hasReasoning && !hasRefusal {
				t.Error("assistant message should have at least content, reasoning_content, or refusal")
			}
		}
	}
	if !found {
		t.Error("expected at least one assistant message")
	}
}

func TestConvertResponsesToChat_OrphanedToolCallWithEmptyReasoning_Dropped(t *testing.T) {
	// Reasoning with empty summary + orphaned function_call: should be dropped
	inputJSON := `[
		{"type":"message","id":"msg_1","role":"user","status":"completed","content":[{"type":"input_text","text":"Hello"}]},
		{"type":"reasoning","id":"r_1","summary":[]},
		{"type":"function_call","id":"fc_1","call_id":"call_orphan","name":"get_weather","arguments":"{}","status":"completed"},
		{"type":"message","id":"msg_2","role":"user","status":"completed","content":[{"type":"input_text","text":"Follow up"}]}
	]`

	respReq := &ResponsesRequest{
		Model: "o3",
		Input: json.RawMessage(inputJSON),
	}

	chatReq, err := ConvertResponsesToChatRequest(respReq)
	if err != nil {
		t.Fatalf("conversion error: %v", err)
	}

	// No assistant message should be present since reasoning was empty and tool_calls orphaned
	for _, msg := range chatReq.Messages {
		if msg.Role == "assistant" {
			t.Errorf("should not have assistant message, but found one with content=%v reasoning=%v tool_calls=%d",
				msg.Content, msg.ReasoningContent, len(msg.ToolCalls))
		}
	}
}
