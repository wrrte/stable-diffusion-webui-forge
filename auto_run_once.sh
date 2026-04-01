#!/bin/bash

# 무한 반복 루프
while true
do
    echo "이미지 생성을 시작합니다..."
    
    # 경로에 맞게 명령어 수정
    ./webui.sh --auto-generate-once
    
    echo "생성 완료 및 웹UI 종료됨. 3초 후 다시 시작합니다..."
    sleep 3
done