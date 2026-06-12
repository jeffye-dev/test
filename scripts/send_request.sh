ROUTER_URL=http://111.4.242.20:30002

    #"model": "minimax-m25-h20e",
    #"model": "/models/MiniMax-M2.5",
    #"stream": true
    #"temperature": 0,

TOTAL=100
#TOTAL=2
for IDX in $(seq 1 $TOTAL); do
echo "=======$IDX======="
curl -v -X POST $ROUTER_URL/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer a3f8b1c7e9d2a5f60b8c3d4e7f9a1b2c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a' \
  -d '{
    "model": "Kimi-K2.5-mix-vllm",
    "messages": [
        {"role": "system", "content": "你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。你是一个专业的 AI 助手。"},
        {"role": "user", "content": "请写出岳飞的诗词满江红全文。"}
    ],
    "top_p": 0.95,
    "max_tokens": 1024
}'
done

