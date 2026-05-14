package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync/atomic"
	"time"
)

// ==================== Chat Completions → Responses API ====================

func ConvertChatToResponsesRequest(chatReq *ChatCompletionsRequest) (*ResponsesRequest, error) {
	out := &ResponsesRequest{
		Model:  chatReq.Model,
		Stream: chatReq.Stream,
	}

	// 1. Convert messages → input and instructions
	inputMsgs, instructions, err := convertChatMessagesToResponsesInput(chatReq.Messages)
	if err != nil {
		return nil, fmt.Errorf("convert messages: %w", err)
	}
	out.Input = json.RawMessage(mustMarshal(inputMsgs))
	if instructions != "" {
		out.Instructions = &instructions
	}

	// 4. max_tokens / max_completion_tokens → max_output_tokens
	if chatReq.MaxCompletionTokens != nil {
		out.MaxOutputTokens = chatReq.MaxCompletionTokens
	} else if chatReq.MaxTokens != nil {
		out.MaxOutputTokens = chatReq.MaxTokens
	}

	// 5. Temperature, top_p, frequency_penalty, presence_penalty — direct assignment
	out.Temperature = chatReq.Temperature
	out.TopP = chatReq.TopP
	out.FrequencyPenalty = chatReq.FrequencyPenalty
	out.PresencePenalty = chatReq.PresencePenalty

	// n parameter warning
	if chatReq.N != nil && *chatReq.N > 1 {
		log.Printf("[chat->resp] WARNING: n=%d requested but Responses API only supports 1 output", *chatReq.N)
	}

	// stop
	out.Stop = chatReq.Stop

	// seed
	out.Seed = chatReq.Seed

	// store
	out.Store = chatReq.Store

	// metadata
	out.Metadata = chatReq.Metadata

	// service_tier
	out.ServiceTier = chatReq.ServiceTier

	// logprobs → top_logprobs
	if chatReq.TopLogprobs != nil {
		out.TopLogprobs = chatReq.TopLogprobs
	} else if chatReq.Logprobs != nil && *chatReq.Logprobs {
		v := 1
		out.TopLogprobs = &v
	}

	// 7. reasoning_effort → reasoning
	if chatReq.ReasoningEffort != nil {
		out.Reasoning = &ResponsesReasoning{
			Effort: *chatReq.ReasoningEffort,
		}
	}

	// response_format → text
	if chatReq.ResponseFormat != nil {
		if text := convertResponseFormatToText(chatReq.ResponseFormat); text != nil {
			out.Text = json.RawMessage(mustMarshal(text))
		}
	}

	// parallel_tool_calls
	out.ParallelToolCalls = chatReq.ParallelToolCalls

	// 6. Convert tools
	if len(chatReq.Tools) > 0 {
		respTools := convertChatToolsToResponses(chatReq.Tools)
		out.Tools = json.RawMessage(mustMarshal(respTools))
	}

	// tool_choice
	out.ToolChoice = chatReq.ToolChoice

	// user
	out.User = chatReq.User

	// 8. stream-specific fields
	if chatReq.Stream {
		out.Include = []string{"reasoning.encrypted_content"}
	}

	// 9. stream_options
	if chatReq.Stream {
		if chatReq.StreamOptions != nil {
			out.StreamOptions = chatReq.StreamOptions
		} else {
			out.StreamOptions = &StreamOptions{IncludeUsage: true}
		}
	}

	return out, nil
}

// ConvertResponsesRespToChatResp converts Responses API response → Chat Completions response
func ConvertResponsesRespToChatResp(respResp *ResponsesResponse) (*ChatCompletionsResponse, error) {
	chatResp := &ChatCompletionsResponse{
		ID:          convertID(respResp.ID, "chatcmpl-"),
		Object:      "chat.completion",
		Created:     respResp.CreatedAt,
		Model:       respResp.Model,
		ServiceTier: respResp.ServiceTier,
	}

	var textParts []string
	var toolCalls []ToolCall
	var refusal *string
	finishReason := "stop"

	for _, item := range respResp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				switch part.Type {
				case "output_text":
					textParts = append(textParts, part.Text)
				case "refusal":
					r := part.Refusal
					if r == "" {
						r = part.Text
					}
					refusal = &r
				}
			}

		case "function_call":
			toolCalls = append(toolCalls, ToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
			finishReason = "tool_calls"
		}
	}

	// Check incomplete_details for length finish reason
	if respResp.Status == "incomplete" {
		finishReason = "length"
	}

	msg := ChatMessage{
		Role:    "assistant",
		Refusal: refusal,
	}
	if len(textParts) > 0 {
		combined := strings.Join(textParts, "")
		msg.Content = jsonString(combined)
	} else {
		msg.Content = json.RawMessage("null")
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	chatResp.Choices = []ChatChoice{
		{
			Index:        0,
			Message:      &msg,
			FinishReason: &finishReason,
		},
	}

	// Convert usage with details
	if respResp.Usage != nil {
		chatResp.Usage = &ChatUsage{
			PromptTokens:     respResp.Usage.InputTokens,
			CompletionTokens: respResp.Usage.OutputTokens,
			TotalTokens:      respResp.Usage.TotalTokens,
		}
		if respResp.Usage.OutputTokensDetails != nil && respResp.Usage.OutputTokensDetails.ReasoningTokens > 0 {
			chatResp.Usage.CompletionTokensDetails = &CompletionTokensDetails{
				ReasoningTokens: respResp.Usage.OutputTokensDetails.ReasoningTokens,
			}
		}
		if respResp.Usage.InputTokensDetails != nil && respResp.Usage.InputTokensDetails.CachedTokens > 0 {
			chatResp.Usage.PromptTokensDetails = &PromptTokensDetails{
				CachedTokens: respResp.Usage.InputTokensDetails.CachedTokens,
			}
		}
	}

	return chatResp, nil
}

// ==================== Responses API → Chat Completions ====================

func ConvertResponsesToChatRequest(respReq *ResponsesRequest) (*ChatCompletionsRequest, error) {
	chatReq := &ChatCompletionsRequest{
		Model:  respReq.Model,
		Stream: respReq.Stream,
	}

	var messages []ChatMessage

	// Add instructions as system message
	if respReq.Instructions != nil && *respReq.Instructions != "" {
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: jsonString(*respReq.Instructions),
		})
	}

	// Parse input
	if respReq.Input != nil {
		var inputStr string
		if err := json.Unmarshal(respReq.Input, &inputStr); err == nil {
			messages = append(messages, ChatMessage{
				Role:    "user",
				Content: jsonString(inputStr),
			})
		} else {
			var inputMsgs []json.RawMessage
			if err := json.Unmarshal(respReq.Input, &inputMsgs); err == nil {
				// Process each input message with type awareness
				var pendingReasoning string  // reasoning text to attach to next assistant message
				var pendingToolCalls []ToolCall // accumulated function_call items

				flushToolCalls := func() {
					if len(pendingToolCalls) == 0 {
						return
					}
					m := ChatMessage{
						Role:      "assistant",
						ToolCalls: pendingToolCalls,
					}
					if pendingReasoning != "" {
						m.ReasoningContent = &pendingReasoning
						pendingReasoning = ""
					}
					messages = append(messages, m)
					pendingToolCalls = nil
				}

				for _, rawMsg := range inputMsgs {
					var im ResponsesInputMessage
					json.Unmarshal(rawMsg, &im)

					switch {
					case im.Type == "reasoning":
						// Extract reasoning text from summary and hold for next assistant message
						if im.Summary != nil {
							var summary []struct {
								Type string `json:"type"`
								Text string `json:"text"`
							}
							if json.Unmarshal(im.Summary, &summary) == nil {
								var parts []string
								for _, s := range summary {
									if s.Text != "" {
										parts = append(parts, s.Text)
									}
								}
								if len(parts) > 0 {
									if pendingReasoning != "" {
										pendingReasoning += "\n" + strings.Join(parts, "\n")
									} else {
										pendingReasoning = strings.Join(parts, "\n")
									}
								}
							}
						}

					case im.Type == "function_call_output":
						flushToolCalls()
						messages = append(messages, ChatMessage{
							Role:       "tool",
							Content:    jsonString(im.Output),
							ToolCallID: im.CallID,
						})

					case im.Type == "function_call":
						pendingToolCalls = append(pendingToolCalls, ToolCall{
							ID:   im.CallID,
							Type: "function",
							Function: FunctionCall{
								Name:      im.Name,
								Arguments: im.Arguments,
							},
						})

					default:
						flushToolCalls()
						role := im.Role
						if role == "" {
							role = "assistant"
						}
						if role == "developer" {
							role = "system"
						}
						m := ChatMessage{
							Role: role,
						}
						if im.Content != nil {
							contentVal := convertResponsesContentToChat(im.Content)
							if contentVal != nil {
								m.Content = json.RawMessage(mustMarshal(contentVal))
							}
							// Extract refusal from content parts
							var parts []ResponsesContentPart
							if err := json.Unmarshal(im.Content, &parts); err == nil {
								for _, p := range parts {
									if p.Type == "refusal" && p.Refusal != "" {
										m.Refusal = &p.Refusal
										break
									}
								}
							}
						}
						// Attach pending reasoning_content to the next assistant message
						if pendingReasoning != "" && role == "assistant" {
							m.ReasoningContent = &pendingReasoning
							pendingReasoning = ""
						}
						messages = append(messages, m)
					}
				}
				flushToolCalls()
			}
		}
	}

	messages = cleanupOrphanedToolCalls(messages)
	chatReq.Messages = messages

	// ---- Parameter mapping ----

	// max_output_tokens → max_completion_tokens
	if respReq.MaxOutputTokens != nil {
		chatReq.MaxCompletionTokens = respReq.MaxOutputTokens
	}
	if respReq.Temperature != nil {
		chatReq.Temperature = respReq.Temperature
	}
	if respReq.TopP != nil {
		chatReq.TopP = respReq.TopP
	}
	if respReq.FrequencyPenalty != nil {
		chatReq.FrequencyPenalty = respReq.FrequencyPenalty
	}
	if respReq.PresencePenalty != nil {
		chatReq.PresencePenalty = respReq.PresencePenalty
	}

	// store
	if respReq.Store != nil {
		chatReq.Store = respReq.Store
	}

	// metadata — direct assignment of json.RawMessage
	if respReq.Metadata != nil {
		chatReq.Metadata = respReq.Metadata
	}

	// service_tier
	if respReq.ServiceTier != nil {
		chatReq.ServiceTier = respReq.ServiceTier
	}

	// top_logprobs → logprobs + top_logprobs
	if respReq.TopLogprobs != nil && *respReq.TopLogprobs > 0 {
		logprobs := true
		chatReq.Logprobs = &logprobs
		chatReq.TopLogprobs = respReq.TopLogprobs
	}

	// reasoning.effort → reasoning_effort
	if respReq.Reasoning != nil && respReq.Reasoning.Effort != "" {
		chatReq.ReasoningEffort = &respReq.Reasoning.Effort
	}

	// text.format → response_format
	if respReq.Text != nil {
		if rf := convertTextToResponseFormat(respReq.Text); rf != nil {
			chatReq.ResponseFormat = json.RawMessage(mustMarshal(rf))
		}
	}

	// parallel_tool_calls
	if respReq.ParallelToolCalls != nil {
		chatReq.ParallelToolCalls = respReq.ParallelToolCalls
	}

	// Convert tools (handle function + skip unsupported types with warning)
	if respReq.Tools != nil {
		var respTools []map[string]interface{}
		if err := json.Unmarshal(respReq.Tools, &respTools); err == nil {
			var chatTools []ChatTool
			for _, rt := range respTools {
				toolType, _ := rt["type"].(string)
				switch toolType {
				case "function":
					name, _ := rt["name"].(string)
					ct := ChatTool{
						Type: "function",
						Function: ChatFunction{
							Name: name,
						},
					}
					if desc, ok := rt["description"].(string); ok {
						ct.Function.Description = desc
					}
					if params, ok := rt["parameters"]; ok {
						// Extract strict from parameters if present (robustness)
						if paramsMap, ok := params.(map[string]interface{}); ok {
							if s, ok := paramsMap["strict"].(bool); ok {
								ct.Function.Strict = &s
								delete(paramsMap, "strict")
							}
						}
						ct.Function.Parameters = json.RawMessage(mustMarshal(params))
					}
					if strict, ok := rt["strict"].(bool); ok {
						ct.Function.Strict = &strict
					}
					chatTools = append(chatTools, ct)
				case "namespace":
					// Flatten namespace tools (e.g., MCP) with prefixed names
					nsName, _ := rt["name"].(string)
					var nsTools []map[string]interface{}
					if raw, ok := rt["tools"]; ok {
						b, _ := json.Marshal(raw)
						json.Unmarshal(b, &nsTools)
					}
					for _, nt := range nsTools {
						ntType, _ := nt["type"].(string)
						if ntType != "function" {
							continue
						}
						ntName, _ := nt["name"].(string)
						ct := ChatTool{
							Type: "function",
							Function: ChatFunction{
								Name: nsName + ntName,
							},
						}
						if desc, ok := nt["description"].(string); ok {
							ct.Function.Description = desc
						}
						if params, ok := nt["parameters"]; ok {
							ct.Function.Parameters = json.RawMessage(mustMarshal(params))
						}
						if strict, ok := nt["strict"].(bool); ok {
							ct.Function.Strict = &strict
						}
						chatTools = append(chatTools, ct)
					}
				case "custom":
					name, _ := rt["name"].(string)
					desc, _ := rt["description"].(string)
					if format, ok := rt["format"].(map[string]interface{}); ok {
						if def, ok := format["definition"].(string); ok && def != "" {
							syntax, _ := format["syntax"].(string)
							desc += "\n\nThis tool uses a structured grammar format (" + syntax + "). " +
								"The argument must follow this grammar:\n" + def
						}
					}
					falseVal := false
					ct := ChatTool{
						Type: "function",
						Function: ChatFunction{
							Name:        name,
							Description: desc,
							Strict:      &falseVal,
							Parameters: json.RawMessage(mustMarshal(map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"input": map[string]interface{}{
										"type":        "string",
										"description": "The patch content in the tool's native grammar format.",
									},
								},
								"required":             []string{"input"},
								"additionalProperties": false,
							})),
						},
					}
					chatTools = append(chatTools, ct)
				default:
					// web_search, file_search, code_interpreter, computer_use
					// cannot be mapped to Chat Completions — silently skip
				}
			}
			if len(chatTools) > 0 {
				chatReq.Tools = chatTools
			}
		}
	}

	// tool_choice — direct assignment of json.RawMessage
	if respReq.ToolChoice != nil {
		chatReq.ToolChoice = respReq.ToolChoice
	}

	if respReq.User != nil {
		chatReq.User = respReq.User
	}

	if respReq.Stream {
		chatReq.StreamOptions = &StreamOptions{IncludeUsage: true}
	}

	// max_tokens default — ensure downstream API has a limit
	if chatReq.MaxTokens == nil && chatReq.MaxCompletionTokens == nil {
		defaultMax := 64000
		chatReq.MaxTokens = &defaultMax
	}

	return chatReq, nil
}

// ConvertChatRespToResponsesResp converts Chat Completions response → Responses API response
func ConvertChatRespToResponsesResp(chatResp *ChatCompletionsResponse) (*ResponsesResponse, error) {
	respResp := &ResponsesResponse{
		ID:          convertID(chatResp.ID, "resp_"),
		Object:      "response",
		CreatedAt:   chatResp.Created,
		Status:      "completed",
		Model:       chatResp.Model,
		ServiceTier: chatResp.ServiceTier,
	}

	for _, choice := range chatResp.Choices {
		if choice.Message == nil {
			continue
		}
		msg := choice.Message

		// Check finish reason for incomplete
		if choice.FinishReason != nil && *choice.FinishReason == "length" {
			respResp.Status = "incomplete"
			respResp.IncompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
		}

		// Add reasoning output item if present (e.g., DeepSeek reasoning_content)
		if msg.ReasoningContent != nil && *msg.ReasoningContent != "" {
			respResp.Output = append(respResp.Output, OutputItem{
				ID:      generateID("rs_"),
				Type:    "reasoning",
				Status:  "completed",
				Summary: []ResponsesSummary{{Type: "summary_text", Text: *msg.ReasoningContent}},
			})
		}

		// Add message output item
		text := contentToString(msg.Content)
		hasText := text != ""
		if hasText || len(msg.ToolCalls) == 0 {
			outputItem := OutputItem{
				ID:     fmt.Sprintf("msg_%d", time.Now().UnixNano()),
				Type:   "message",
				Status: "completed",
				Role:   "assistant",
				Phase:  "final_answer",
			}

			if msg.Refusal != nil && *msg.Refusal != "" {
				outputItem.Content = []ContentPart{
					{
						Type:    "refusal",
						Refusal: *msg.Refusal,
					},
				}
			} else {
				outputItem.Content = []ContentPart{
					{
						Type:        "output_text",
						Text:        text,
						Annotations: json.RawMessage("[]"),
					},
				}
			}

			respResp.Output = append(respResp.Output, outputItem)
		}

		// Add function_call output items for tool calls
		for _, tc := range msg.ToolCalls {
			outputItem := OutputItem{
				ID:        tc.ID,
				Type:      "function_call",
				Status:    "completed",
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				CallID:    tc.ID,
			}
			respResp.Output = append(respResp.Output, outputItem)
		}
	}

	// Convert usage with details
	if chatResp.Usage != nil {
		respResp.Usage = &ResponsesUsage{
			InputTokens:  chatResp.Usage.PromptTokens,
			OutputTokens: chatResp.Usage.CompletionTokens,
			TotalTokens:  chatResp.Usage.TotalTokens,
		}
		if chatResp.Usage.CompletionTokensDetails != nil {
			respResp.Usage.OutputTokensDetails = &OutputTokensDetails{
				ReasoningTokens: chatResp.Usage.CompletionTokensDetails.ReasoningTokens,
			}
		}
		if chatResp.Usage.PromptTokensDetails != nil {
			respResp.Usage.InputTokensDetails = &InputTokensDetails{
				CachedTokens: chatResp.Usage.PromptTokensDetails.CachedTokens,
			}
		}
	}

	return respResp, nil
}

// ==================== Streaming State Machine ====================

// ResponsesEventToChatState tracks state for converting Responses SSE events to Chat chunks.
type ResponsesEventToChatState struct {
	ID      string
	Model   string
	Created int64

	SentRole    bool
	SawToolCall bool
	SawText     bool
	Finalized   bool

	NextToolCallIndex      int
	OutputIndexToToolIndex map[int]int

	IncludeUsage bool
	Usage        *ChatUsage
}

func NewResponsesEventToChatState() *ResponsesEventToChatState {
	return &ResponsesEventToChatState{
		ID:                     generateID("chatcmpl-"),
		Created:                nowUnix(),
		OutputIndexToToolIndex: make(map[int]int),
		IncludeUsage:           true,
	}
}

// ResponsesEventToChatChunks converts a single Responses SSE event into Chat chunks.
func ResponsesEventToChatChunks(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	switch evt.Type {
	case "response.created":
		return resToChatHandleCreated(evt, state)
	case "response.output_text.delta":
		return resToChatHandleTextDelta(evt, state)
	case "response.content_part.delta":
		return resToChatHandleTextDelta(evt, state)
	case "response.refusal.delta":
		return resToChatHandleRefusalDelta(evt, state)
	case "response.output_item.added":
		return resToChatHandleOutputItemAdded(evt, state)
	case "response.function_call_arguments.delta":
		return resToChatHandleFuncArgsDelta(evt, state)
	case "response.reasoning_summary_text.delta":
		return resToChatHandleReasoningDelta(evt, state)
	case "response.reasoning_summary_text.done":
		return nil
	case "response.completed", "response.done", "response.incomplete", "response.failed":
		return resToChatHandleCompleted(evt, state)
	default:
		return nil
	}
}

// FinalizeResponsesChatStream emits a final chunk if stream ended without completion event.
func FinalizeResponsesChatStream(state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if state.Finalized {
		return nil
	}
	state.Finalized = true

	finishReason := "stop"
	if state.SawToolCall {
		finishReason = "tool_calls"
	}

	chunks := []ChatCompletionsChunk{makeChatFinishChunk(state, finishReason)}

	if state.IncludeUsage && state.Usage != nil {
		chunks = append(chunks, ChatCompletionsChunk{
			ID: state.ID, Object: "chat.completion.chunk",
			Created: state.Created, Model: state.Model,
			Choices: []ChatChunkChoice{},
			Usage:   state.Usage,
		})
	}

	return chunks
}

func resToChatHandleCreated(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Response != nil {
		if evt.Response.ID != "" {
			state.ID = evt.Response.ID
		}
		if state.Model == "" && evt.Response.Model != "" {
			state.Model = evt.Response.Model
		}
	}
	if state.SentRole {
		return nil
	}
	state.SentRole = true
	role := "assistant"
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{Role: role})}
}

func resToChatHandleTextDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}
	state.SawText = true
	content := evt.Delta
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{Content: &content})}
}

func resToChatHandleRefusalDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}
	refusal := evt.Delta
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{Refusal: &refusal})}
}

func resToChatHandleOutputItemAdded(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Item == nil || evt.Item.Type != "function_call" {
		return nil
	}
	state.SawToolCall = true
	idx := state.NextToolCallIndex
	state.OutputIndexToToolIndex[evt.OutputIndex] = idx
	state.NextToolCallIndex++

	callID := evt.Item.CallID
	if callID == "" {
		callID = evt.ItemID
	}
	if callID == "" {
		callID = generateID("call_")
	}

	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{
		ToolCalls: []ToolCall{{
			Index: &idx,
			ID:    callID,
			Type:  "function",
			Function: FunctionCall{Name: evt.Item.Name},
		}},
	})}
}

func resToChatHandleFuncArgsDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}
	idx, ok := state.OutputIndexToToolIndex[evt.OutputIndex]
	if !ok {
		return nil
	}
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{
		ToolCalls: []ToolCall{{
			Index: &idx,
			Function: FunctionCall{Arguments: evt.Delta},
		}},
	})}
}

func resToChatHandleReasoningDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}
	reasoning := evt.Delta
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{ReasoningContent: &reasoning})}
}

func resToChatHandleCompleted(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	state.Finalized = true
	finishReason := "stop"

	if evt.Response != nil {
		if evt.Response.Usage != nil {
			u := evt.Response.Usage
			usage := &ChatUsage{
				PromptTokens:     u.InputTokens,
				CompletionTokens: u.OutputTokens,
				TotalTokens:      u.InputTokens + u.OutputTokens,
			}
			if u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0 {
				usage.PromptTokensDetails = &PromptTokensDetails{CachedTokens: u.InputTokensDetails.CachedTokens}
			}
			if u.OutputTokensDetails != nil && u.OutputTokensDetails.ReasoningTokens > 0 {
				usage.CompletionTokensDetails = &CompletionTokensDetails{
					ReasoningTokens: u.OutputTokensDetails.ReasoningTokens,
				}
			}
			state.Usage = usage
		}
		switch evt.Response.Status {
		case "incomplete":
			if evt.Response.IncompleteDetails != nil && evt.Response.IncompleteDetails.Reason == "max_output_tokens" {
				finishReason = "length"
			}
		case "completed":
			if state.SawToolCall {
				finishReason = "tool_calls"
			}
		}
	} else if state.SawToolCall {
		finishReason = "tool_calls"
	}

	var chunks []ChatCompletionsChunk
	chunks = append(chunks, makeChatFinishChunk(state, finishReason))

	if state.IncludeUsage && state.Usage != nil {
		chunks = append(chunks, ChatCompletionsChunk{
			ID: state.ID, Object: "chat.completion.chunk",
			Created: state.Created, Model: state.Model,
			Choices: []ChatChunkChoice{},
			Usage:   state.Usage,
		})
	}

	return chunks
}

func makeChatDeltaChunk(state *ResponsesEventToChatState, delta ChatDelta) ChatCompletionsChunk {
	return ChatCompletionsChunk{
		ID: state.ID, Object: "chat.completion.chunk",
		Created: state.Created, Model: state.Model,
		Choices: []ChatChunkChoice{{Index: 0, Delta: delta}},
	}
}

func makeChatFinishChunk(state *ResponsesEventToChatState, finishReason string) ChatCompletionsChunk {
	empty := ""
	return ChatCompletionsChunk{
		ID: state.ID, Object: "chat.completion.chunk",
		Created: state.Created, Model: state.Model,
		Choices: []ChatChunkChoice{{
			Index: 0,
			Delta: ChatDelta{Content: &empty},
			FinishReason: &finishReason,
		}},
	}
}

// ChatChunkToSSE formats a chunk as SSE data line.
func ChatChunkToSSE(chunk ChatCompletionsChunk) (string, error) {
	data, err := json.Marshal(chunk)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("data: %s\n\n", data), nil
}

// ==================== Vision Content Conversion ====================

// convertChatContentToResponses converts Chat Completions content (string or multipart array)
// to Responses API input format, handling image_url → input_image conversion
func convertChatContentToResponses(raw json.RawMessage) interface{} {
	if raw == nil {
		return nil
	}

	// Try as string
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	// Try as multipart array
	var parts []ChatContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		var result []map[string]interface{}
		for _, p := range parts {
			switch p.Type {
			case "text":
				result = append(result, map[string]interface{}{
					"type": "input_text",
					"text": p.Text,
				})
			case "image_url":
				if p.ImageURL != nil && p.ImageURL.URL != "" && !isEmptyBase64DataURI(p.ImageURL.URL) {
					img := map[string]interface{}{
						"type":      "input_image",
						"image_url": p.ImageURL.URL,
					}
					if p.ImageURL.Detail != "" {
						img["detail"] = p.ImageURL.Detail
					}
					result = append(result, img)
				}
			default:
				// Pass through unknown types
				var raw map[string]interface{}
				b, _ := json.Marshal(p)
				json.Unmarshal(b, &raw)
				result = append(result, raw)
			}
		}
		return result
	}

	// Fallback
	var raw2 interface{}
	json.Unmarshal(raw, &raw2)
	return raw2
}

// convertResponsesContentToChat converts Responses API content (string or multipart array)
// to Chat Completions format, handling input_image → image_url conversion
func convertResponsesContentToChat(raw json.RawMessage) interface{} {
	if raw == nil {
		return nil
	}

	// Try as string
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	// Try as multipart array
	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		var result []map[string]interface{}
		for _, p := range parts {
			switch p.Type {
			case "input_text", "output_text", "text":
				result = append(result, map[string]interface{}{
					"type": "text",
					"text": p.Text,
				})
			case "input_image":
				img := map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": p.ImageURL,
					},
				}
				if p.Detail != "" {
					img["image_url"].(map[string]interface{})["detail"] = p.Detail
				}
				result = append(result, img)
				case "refusal":
					refusalText := p.Refusal
					if refusalText == "" {
						refusalText = p.Text
					}
					result = append(result, map[string]interface{}{
						"type":    "refusal",
						"refusal": refusalText,
					})
				default:
					var unknown map[string]interface{}
					b, _ := json.Marshal(p)
					json.Unmarshal(b, &unknown)
					result = append(result, unknown)
			}
		}
		return result
	}

	var raw2 interface{}
	json.Unmarshal(raw, &raw2)
	return raw2
}

// ==================== Structured Output Conversion ====================

// convertResponseFormatToText converts Chat Completions response_format → Responses text.format
func convertResponseFormatToText(rf json.RawMessage) map[string]interface{} {
	var rfObj ResponseFormatObj
	if err := json.Unmarshal(rf, &rfObj); err != nil {
		return nil
	}
	if rfObj.Type == "" {
		return nil
	}

	result := map[string]interface{}{}

	switch rfObj.Type {
	case "json_object":
		result["format"] = map[string]interface{}{
			"type": "json_object",
		}
	case "json_schema":
		if rfObj.JSONSchema != nil {
			format := map[string]interface{}{
				"type": "json_schema",
				"name": rfObj.JSONSchema.Name,
			}
			if rfObj.JSONSchema.Description != "" {
				format["description"] = rfObj.JSONSchema.Description
			}
			if rfObj.JSONSchema.Schema != nil {
				var s interface{}
				json.Unmarshal(rfObj.JSONSchema.Schema, &s)
				format["schema"] = s
			}
			if rfObj.JSONSchema.Strict != nil {
				format["strict"] = *rfObj.JSONSchema.Strict
			}
			result["format"] = format
		}
	case "text":
		result["format"] = map[string]interface{}{
			"type": "text",
		}
	default:
		// Pass through
		var raw interface{}
		json.Unmarshal(rf, &raw)
		result["format"] = raw
	}

	return result
}

// convertTextToResponseFormat converts Responses text.format → Chat Completions response_format
func convertTextToResponseFormat(text json.RawMessage) interface{} {
	var tf ResponsesTextFormat
	if err := json.Unmarshal(text, &tf); err != nil {
		return nil
	}

	// If no format specified (e.g., only verbosity), skip response_format
	if tf.Format.Type == "" {
		return nil
	}

	switch tf.Format.Type {
	case "json_object":
		return map[string]interface{}{
			"type": "json_object",
		}
	case "json_schema":
		result := map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name": tf.Format.Name,
			},
		}
		js := result["json_schema"].(map[string]interface{})
		if tf.Format.Description != "" {
			js["description"] = tf.Format.Description
		}
		if tf.Format.Schema != nil {
			var s interface{}
			json.Unmarshal(tf.Format.Schema, &s)
			js["schema"] = s
		}
		if tf.Format.Strict != nil {
			js["strict"] = *tf.Format.Strict
		}
		return result
	case "text":
		return map[string]interface{}{
			"type": "text",
		}
	default:
		return map[string]interface{}{
			"type": tf.Format.Type,
		}
	}
}

// parseAssistantContent returns assistant content as plain text.
// For structured thinking/reasoning parts, wraps in <thinking>...</thinking> tags.
func parseAssistantContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		return contentToString(raw)
	}

	var b strings.Builder
	for _, p := range parts {
		typ, _ := p["type"].(string)
		text, _ := p["text"].(string)
		thinking, _ := p["thinking"].(string)

		switch typ {
		case "thinking", "reasoning":
			content := thinking
			if content == "" {
				content = text
			}
			if content != "" {
				b.WriteString("<thinking>")
				b.WriteString(content)
				b.WriteString("</thinking>")
			}
		default:
			if text != "" {
				b.WriteString(text)
			}
		}
	}

	return b.String()
}

func isEmptyBase64DataURI(raw string) bool {
	if !strings.HasPrefix(raw, "data:") {
		return false
	}
	rest := strings.TrimPrefix(raw, "data:")
	semicolonIdx := strings.Index(rest, ";")
	if semicolonIdx < 0 {
		return false
	}
	rest = rest[semicolonIdx+1:]
	if !strings.HasPrefix(rest, "base64,") {
		return false
	}
	return strings.TrimSpace(strings.TrimPrefix(rest, "base64,")) == ""
}

func responsesStatusToChatFinishReason(status string, details *ResponsesIncompleteDetails, toolCalls []ToolCall) string {
	switch status {
	case "incomplete":
		if details != nil && details.Reason == "max_output_tokens" {
			return "length"
		}
		return "stop"
	case "completed":
		if len(toolCalls) > 0 {
			return "tool_calls"
		}
		return "stop"
	default:
		return "stop"
	}
}

// convertChatMessagesToResponsesInput converts messages to Responses input items.
// System/developer messages are collected into instructions.
func convertChatMessagesToResponsesInput(msgs []ChatMessage) ([]ResponsesInputMessage, string, error) {
	var out []ResponsesInputMessage
	var instructionsParts []string

	for _, m := range msgs {
		switch m.Role {
		case "system", "developer":
			text := contentToString(m.Content)
			if text != "" {
				instructionsParts = append(instructionsParts, text)
			}
		default:
			items, err := chatMessageToResponsesItems(m)
			if err != nil {
				return nil, "", err
			}
			out = append(out, items...)
		}
	}

	instructions := strings.Join(instructionsParts, "\n\n")
	return out, instructions, nil
}

func chatMessageToResponsesItems(m ChatMessage) ([]ResponsesInputMessage, error) {
	switch m.Role {
	case "user":
		return chatUserToResponses(m)
	case "assistant":
		return chatAssistantToResponses(m)
	case "tool":
		return chatToolToResponses(m)
	case "function":
		return chatFunctionToResponses(m)
	default:
		return chatUserToResponses(m)
	}
}

func chatUserToResponses(m ChatMessage) ([]ResponsesInputMessage, error) {
	if m.Content != nil {
		return []ResponsesInputMessage{{
			Role:    "user",
			Content: json.RawMessage(mustMarshal(convertChatContentToResponses(m.Content))),
		}}, nil
	}
	return []ResponsesInputMessage{{Role: "user"}}, nil
}

func chatAssistantToResponses(m ChatMessage) ([]ResponsesInputMessage, error) {
	var items []ResponsesInputMessage

	if len(m.Content) > 0 {
		s := parseAssistantContent(m.Content)
		if s != "" {
			parts := []map[string]interface{}{
				{"type": "output_text", "text": s, "annotations": []interface{}{}},
			}
			items = append(items, ResponsesInputMessage{
				Role:    "assistant",
				Content: json.RawMessage(mustMarshal(parts)),
			})
		}
	}

	for _, tc := range m.ToolCalls {
		args := tc.Function.Arguments
		if args == "" {
			args = "{}"
		}
		items = append(items, ResponsesInputMessage{
			Type:      "function_call",
			ID:        tc.ID,
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
			Status:    "completed",
		})
	}

	return items, nil
}

func chatToolToResponses(m ChatMessage) ([]ResponsesInputMessage, error) {
	output := contentToString(m.Content)
	if output == "" {
		output = "(empty)"
	}
	return []ResponsesInputMessage{{
		Type:   "function_call_output",
		CallID: m.ToolCallID,
		Output: output,
	}}, nil
}

func chatFunctionToResponses(m ChatMessage) ([]ResponsesInputMessage, error) {
	output := contentToString(m.Content)
	if output == "" {
		output = "(empty)"
	}
	return []ResponsesInputMessage{{
		Type:   "function_call_output",
		CallID: m.Name,
		Output: output,
	}}, nil
}

func convertChatToolsToResponses(tools []ChatTool) []ResponsesTool {
	var out []ResponsesTool
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		rt := ResponsesTool{
			Type: "function",
			Name: t.Function.Name,
		}
		if t.Function.Description != "" {
			rt.Description = t.Function.Description
		}
		if t.Function.Parameters != nil {
			rt.Parameters = t.Function.Parameters
		}
		if t.Function.Strict != nil {
			rt.Strict = t.Function.Strict
		}
		out = append(out, rt)
	}
	return out
}

// ==================== Helpers ====================

// cleanupOrphanedToolCalls removes tool_calls from assistant messages that lack
// matching tool result messages, and removes tool result messages that reference
// tool_calls no longer present. This prevents upstream API 400 errors when
// conversation history has been compacted and tool results were lost.
func cleanupOrphanedToolCalls(messages []ChatMessage) []ChatMessage {
	// Collect all tool_call_ids that have matching tool results.
	toolResultIDs := make(map[string]bool)
	for _, m := range messages {
		if m.Role == "tool" && m.ToolCallID != "" {
			toolResultIDs[m.ToolCallID] = true
		}
	}

	cleaned := make([]ChatMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role != "assistant" || len(m.ToolCalls) == 0 {
			cleaned = append(cleaned, m)
			continue
		}

		// Keep only tool_calls that have a matching result.
		var validCalls []ToolCall
		for _, tc := range m.ToolCalls {
			if tc.ID != "" && toolResultIDs[tc.ID] {
				validCalls = append(validCalls, tc)
			}
		}

		if len(validCalls) > 0 {
			m.ToolCalls = validCalls
			cleaned = append(cleaned, m)
			continue
		}

		// All tool_calls are orphaned. Keep the message only if it has
		// text content or reasoning content; otherwise drop it entirely.
		hasContent := m.Content != nil && string(m.Content) != "null" && string(m.Content) != "\"\""
		hasReasoning := m.ReasoningContent != nil && *m.ReasoningContent != ""

		if hasContent || hasReasoning {
			m.ToolCalls = nil
			cleaned = append(cleaned, m)
		}
		// else: drop the empty assistant message entirely
	}

	// Reverse pass: remove tool result messages whose tool_call_id is no
	// longer referenced by any remaining assistant message's tool_calls.
	validIDs := make(map[string]bool)
	for _, m := range cleaned {
		for _, tc := range m.ToolCalls {
			if tc.ID != "" {
				validIDs[tc.ID] = true
			}
		}
	}
	var final []ChatMessage
	for _, m := range cleaned {
		if m.Role == "tool" && m.ToolCallID != "" && !validIDs[m.ToolCallID] {
			continue // drop orphaned tool result
		}
		final = append(final, m)
	}

	return final
}

func convertID(id, prefix string) string {
	if strings.HasPrefix(id, prefix) {
		return id
	}
	for _, p := range []string{"chatcmpl-", "resp_", "cmpl-"} {
		if strings.HasPrefix(id, p) {
			return prefix + id[len(p):]
		}
	}
	return prefix + id
}

var idCounter uint64

func generateID(prefix string) string {
	count := atomic.AddUint64(&idCounter, 1)
	return fmt.Sprintf("%s%d_%d", prefix, time.Now().UnixNano(), count)
}
