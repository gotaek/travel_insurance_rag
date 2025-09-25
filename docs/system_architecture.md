graph TD
    subgraph "새로운 RAG 파이프라인 (품질 평가 및 재검색 루프 포함)"
        start([사용자 질문]) --> planner[Planner Node<br/>질문 분석 및 분기 결정]
        
        %% 1. Planner에서 분기
        planner --> |needs_web = true| websearch[Web Search Node<br/>Tavily API 실시간 검색]
        planner --> |needs_web = false| search[Search Node<br/>하이브리드 검색]
        
        %% 2. 검색 및 랭크
        websearch --> search
        search --> rank[Rank Filter Node<br/>스코어 기반 정렬]
        
        %% 3. 문서 검증 및 인용 생성
        rank --> verify[Verify Refine Node<br/>문서 검증 및 인용 생성]
        
        %% 4. 의도에 따른 답변 노드 분기
        subgraph "답변 노드"
            verify --> |intent = qa| qa[QA Node<br/>질문-답변]
            verify --> |intent = summary| summary[Summary Node<br/>문서 요약]
            verify --> |intent = compare| compare[Compare Node<br/>보험사 비교]
            verify --> |intent = recommend| recommend[Recommend Node<br/>추천]
        end
        
        %% 5. 답변 품질 평가 및 개선 루프 (새로 추가)
        qa --> reevaluate[Re-evaluate Node<br/>LLM 기반 답변 품질 평가]
        summary --> reevaluate
        compare --> reevaluate
        recommend --> reevaluate
        
        %% 6. 품질 기반 분기 (새로 추가)
        reevaluate --> |품질 좋음<br/>score >= 0.7| final_answer([최종 답변])
        reevaluate --> |품질 낮음<br/>score < 0.7| replan[Re-plan Node<br/>재검색 질문 생성]
        
        %% 7. 재검색 루프 (새로 추가)
        replan --> planner
        
        %% 스타일링
        classDef newNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
        classDef existingNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
        classDef decisionNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
        
        class reevaluate,replan newNode
        class planner,websearch,search,rank,verify,qa,summary,compare,recommend existingNode
        class final_answer decisionNode
    end
    
    %% 데이터 저장소 및 외부 서비스
    search -.-> vdb[(Vector DB<br/>Chroma)]
    vdb -.-> docs[(PDF 약관)]
    websearch -.-> tavily[Tavily API]
    planner -.-> gemini[Google Gemini<br/>LLM]
    qa-.-> gemini
    summary-.-> gemini
    compare-.-> gemini
    recommend-.-> gemini
    reevaluate-.-> gemini
    replan -.-> gemini