graph TB
    subgraph "사용자 인터페이스"
        UI[Streamlit UI<br/>포트 8501]
    end
    
    subgraph "API 서버"
        API[FastAPI<br/>포트 8000]
        RAG[RAG Router<br/>/rag/ask]
    end
    
    subgraph "LangGraph 워크플로우"
        PLANNER[Planner Node<br/>LLM 기반 분기 결정]
        WEBSEARCH[Web Search Node<br/>실시간 정보 검색]
        SEARCH[Search Node<br/>하이브리드 검색]
        RANK[Rank Filter Node<br/>결과 정렬]
        VERIFY[Verify Refine Node<br/>검증 및 정제]
        
        subgraph "Answerer Nodes"
            QA[QA Node<br/>질문-답변]
            SUMMARY[Summary Node<br/>문서 요약]
            COMPARE[Compare Node<br/>비교 분석]
            RECOMMEND[Recommend Node<br/>추천]
        end
    end
    
    subgraph "검색 엔진"
        VECTOR[Vector Search<br/>FAISS + BGE-M3-SMALL]
        KEYWORD[Keyword Search<br/>BM25]
        HYBRID[Hybrid Search<br/>가중치 병합]
    end
    
    subgraph "데이터 저장소"
        VDB[(Vector DB<br/>index.faiss + index.pkl)]
        DOCS[(Documents<br/>PDF 약관)]
    end
    
    subgraph "외부 서비스"
        GEMINI[Google Gemini<br/>LLM]
        LANGCHAIN[LangSmith<br/>추적 및 모니터링]
    end
    
    %% 사용자 플로우
    UI --> API
    API --> RAG
    RAG --> PLANNER
    
    %% 워크플로우
    PLANNER --> WEBSEARCH
    PLANNER --> SEARCH
    WEBSEARCH --> SEARCH
    SEARCH --> RANK
    RANK --> VERIFY
    VERIFY --> QA
    VERIFY --> SUMMARY
    VERIFY --> COMPARE
    VERIFY --> RECOMMEND
    
    %% 검색 엔진 연결
    SEARCH --> VECTOR
    SEARCH --> KEYWORD
    VECTOR --> HYBRID
    KEYWORD --> HYBRID
    
    %% 데이터 연결
    VECTOR --> VDB
    KEYWORD --> VDB
    VDB --> DOCS
    
    %% LLM 연결
    PLANNER --> GEMINI
    QA --> GEMINI
    SUMMARY --> GEMINI
    COMPARE --> GEMINI
    RECOMMEND --> GEMINI
    
    %% 모니터링
    PLANNER -.-> LANGCHAIN
    QA -.-> LANGCHAIN
    SUMMARY -.-> LANGCHAIN
    COMPARE -.-> LANGCHAIN
    RECOMMEND -.-> LANGCHAIN
    
    %% 스타일링
    classDef uiClass fill:#e1f5fe
    classDef apiClass fill:#f3e5f5
    classDef workflowClass fill:#e8f5e8
    classDef searchClass fill:#fff3e0
    classDef dataClass fill:#fce4ec
    classDef externalClass fill:#f1f8e9
    
    class UI uiClass
    class API,RAG apiClass
    class PLANNER,WEBSEARCH,SEARCH,RANK,VERIFY,QA,SUMMARY,COMPARE,RECOMMEND workflowClass
    class VECTOR,KEYWORD,HYBRID searchClass
    class VDB,DOCS dataClass
    class GEMINI,LANGCHAIN externalClass