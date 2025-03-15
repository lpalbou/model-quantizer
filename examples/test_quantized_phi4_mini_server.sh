#!/bin/bash
# Test script for the Phi4MiniServer with quantized model

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing Phi4MiniServer with quantized model...${NC}"

# Check if the server is running
echo -e "${YELLOW}Checking if server is running...${NC}"
HEALTH_RESPONSE=$(curl -s http://127.0.0.1:8000/health)

if [[ $HEALTH_RESPONSE == *"ready"* ]]; then
    echo -e "${GREEN}Server is running and ready!${NC}"
else
    echo -e "${RED}Server is not running or not ready. Please start the server first.${NC}"
    exit 1
fi

# Check memory usage
echo -e "${YELLOW}Checking memory usage...${NC}"
MEMORY_USAGE=$(ps -o pid,rss,command | grep "python -m system.llm_server.phi4_mini_server" | awk '{print $2/1024/1024 " GB"}' | head -1)
echo -e "${GREEN}Current memory usage: ${MEMORY_USAGE}${NC}"

# Test generation
echo -e "${YELLOW}Testing text generation...${NC}"
PROMPT="What is the capital of France?"
SYSTEM_PROMPT="You are a helpful AI assistant."

echo -e "${YELLOW}Sending request with prompt: '${PROMPT}'${NC}"

RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${PROMPT}\", \"system_prompt\": \"${SYSTEM_PROMPT}\", \"max_tokens\": 100, \"temperature\": 0.2}")

if [[ $RESPONSE == *"text"* ]]; then
    echo -e "${GREEN}Generation successful!${NC}"
    echo -e "${GREEN}Response: ${RESPONSE}${NC}"
    
    # Extract just the generated text
    GENERATED_TEXT=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['text'])")
    echo -e "${GREEN}Generated text: ${GENERATED_TEXT}${NC}"
else
    echo -e "${RED}Generation failed. Response: ${RESPONSE}${NC}"
    exit 1
fi

# Check memory usage after generation
echo -e "${YELLOW}Checking memory usage after generation...${NC}"
MEMORY_USAGE_AFTER=$(ps -o pid,rss,command | grep "python -m system.llm_server.phi4_mini_server" | awk '{print $2/1024/1024 " GB"}' | head -1)
echo -e "${GREEN}Memory usage after generation: ${MEMORY_USAGE_AFTER}${NC}"

echo -e "${GREEN}All tests passed successfully!${NC}" 