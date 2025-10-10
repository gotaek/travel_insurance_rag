export type AskAnswerPayload = {
  conclusion?: string;
  evidence?: string[];
  caveats?: string[];
  comparison_table?: {
    headers: string[];
    rows: string[][];
  };
};

export type AskResponse = {
  final_answer?: AskAnswerPayload;
  draft_answer?: AskAnswerPayload;
  quality_score?: number;
};

export type CreateSessionResponse = {
  session_id: string;
  user_id?: string | null;
  created_at?: string;
  status?: string;
};

export type ExtendedChatMessage = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: number;
  response?: AskResponse; // API 응답 전체 데이터 포함
};


