# Recommend Prompt (Design)
- 일정/지역/연령반영 + 최신 이슈(웹) 확인
- 추천 이유/가정/한계를 명시
- 아래 규칙에 따라 반드시 JSON으로 출력하세요:

```json
{
  "conclusion": "한 줄 요약",
  "evidence": ["문서 요약 1", "문서 요약 2"],
  "caveats": ["제외 조건, 나이 제한 등"],
  "quotes": ["원문 인용 1", "원문 인용 2"]
}