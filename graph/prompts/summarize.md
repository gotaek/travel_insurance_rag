# Summarize Prompt (Design)
- 약관/표를 초등학생도 이해할 표현으로 요약
- 숫자(보장한도/대기기간)는 원문 그대로 유지
- 아래 규칙에 따라 반드시 JSON으로 출력하세요:

```json
{
  "conclusion": "한 줄 요약",
  "evidence": ["문서 요약 1", "문서 요약 2"],
  "caveats": ["제외 조건, 나이 제한 등"],
  "quotes": ["원문 인용 1", "원문 인용 2"]
}