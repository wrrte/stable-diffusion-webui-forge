#!/bin/bash

PROMPT_FILE="/Users/choemj/Desktop/prompts.txt"

# 무한 반복 루프
while true
do
    # 파일이 존재하면서, 공백 문자 외에는 아무런 내용이 없다면(모두 지워졌다면) 루프를 종료합니다.
    if [ -f "$PROMPT_FILE" ] && ! grep -q '[^[:space:]]' "$PROMPT_FILE"; then
        echo "$PROMPT_FILE 에 남은 프롬프트가 없습니다. 스크립트를 종료합니다."
        break
    fi

    echo "이미지 생성을 시작합니다..."
    
    # 경로에 맞게 명령어 수정
    ./webui.sh --auto-generate-once
    
    echo "생성 완료 및 프로그램 종료됨. 20초 후 다시 시작합니다..."
    sleep 20
done

./webui.sh --auto-generate