#!/usr/bin/env python3
"""
멀티턴 대화 및 캐싱 기능 테스트 스크립트
"""

import requests
import json
import time
from typing import Dict, Any


class MultiTurnTester:
    """멀티턴 대화 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        print("🧪 기본 기능 테스트 시작...")
        
        # 1. 헬스 체크
        try:
            response = requests.get(f"{self.base_url}/healthz")
            if response.status_code == 200:
                print("✅ 헬스 체크 통과")
            else:
                print("❌ 헬스 체크 실패")
                return False
        except Exception as e:
            print(f"❌ 헬스 체크 오류: {e}")
            return False
        
        # 2. 캐시 통계 확인
        try:
            response = requests.get(f"{self.base_url}/rag/cache/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ 캐시 통계 조회 성공: {stats}")
            else:
                print("❌ 캐시 통계 조회 실패")
        except Exception as e:
            print(f"⚠️ 캐시 통계 조회 오류: {e}")
        
        return True
    
    def test_session_management(self):
        """세션 관리 테스트"""
        print("\n🧪 세션 관리 테스트 시작...")
        
        # 1. 세션 생성
        try:
            response = requests.post(f"{self.base_url}/rag/session/create", 
                                   params={"user_id": "test_user"})
            if response.status_code == 200:
                session_data = response.json()
                self.session_id = session_data["session_id"]
                print(f"✅ 세션 생성 성공: {self.session_id}")
            else:
                print("❌ 세션 생성 실패")
                return False
        except Exception as e:
            print(f"❌ 세션 생성 오류: {e}")
            return False
        
        # 2. 세션 정보 조회
        try:
            response = requests.get(f"{self.base_url}/rag/session/{self.session_id}")
            if response.status_code == 200:
                session_info = response.json()
                print(f"✅ 세션 정보 조회 성공: {session_info}")
            else:
                print("❌ 세션 정보 조회 실패")
        except Exception as e:
            print(f"❌ 세션 정보 조회 오류: {e}")
        
        return True
    
    def test_multiturn_conversation(self):
        """멀티턴 대화 테스트"""
        if not self.session_id:
            print("❌ 세션이 없어 멀티턴 테스트를 건너뜁니다")
            return False
        
        print(f"\n🧪 멀티턴 대화 테스트 시작 (세션: {self.session_id})...")
        
        # 대화 시나리오
        questions = [
            "여행자보험에 대해 알려주세요",
            "그 보험의 보상 한도는 얼마인가요?",
            "해외여행 시 특별히 주의할 점이 있나요?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- 질문 {i}: {question} ---")
            
            try:
                response = requests.post(
                    f"{self.base_url}/rag/multiturn/ask",
                    json={
                        "question": question,
                        "session_id": self.session_id,
                        "user_id": "test_user"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 답변 {i} 성공")
                    print(f"   의도: {result.get('intent', 'unknown')}")
                    print(f"   답변: {result.get('draft_answer', {}).get('conclusion', '')[:100]}...")
                    
                    # 컨텍스트 정보 확인
                    if "conversation_context" in result:
                        context = result["conversation_context"]
                        print(f"   턴 수: {context.get('turn_count', 0)}")
                        print(f"   토큰 수: {context.get('total_tokens', 0)}")
                else:
                    print(f"❌ 답변 {i} 실패: {response.status_code}")
                    print(f"   오류: {response.text}")
                
                # 요청 간 간격
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ 답변 {i} 오류: {e}")
        
        return True
    
    def test_caching_performance(self):
        """캐싱 성능 테스트"""
        print("\n🧪 캐싱 성능 테스트 시작...")
        
        test_question = "여행자보험 청구 절차는 어떻게 되나요?"
        
        # 첫 번째 요청 (캐시 미스)
        print("첫 번째 요청 (캐시 미스)...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/rag/ask",
                json={"question": test_question}
            )
            first_duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ 첫 번째 요청 성공: {first_duration:.2f}초")
            else:
                print(f"❌ 첫 번째 요청 실패: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 첫 번째 요청 오류: {e}")
            return False
        
        # 두 번째 요청 (캐시 히트)
        print("두 번째 요청 (캐시 히트)...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/rag/ask",
                json={"question": test_question}
            )
            second_duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ 두 번째 요청 성공: {second_duration:.2f}초")
                
                # 성능 개선 확인
                if second_duration < first_duration:
                    improvement = ((first_duration - second_duration) / first_duration) * 100
                    print(f"🚀 성능 개선: {improvement:.1f}% 빨라짐")
                else:
                    print("⚠️ 캐시 효과가 제한적입니다")
            else:
                print(f"❌ 두 번째 요청 실패: {response.status_code}")
        except Exception as e:
            print(f"❌ 두 번째 요청 오류: {e}")
        
        return True
    
    def test_backward_compatibility(self):
        """하위 호환성 테스트"""
        print("\n🧪 하위 호환성 테스트 시작...")
        
        # 기존 API 형식으로 요청
        try:
            response = requests.post(
                f"{self.base_url}/rag/ask",
                json={"question": "기존 API 호환성 테스트"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 필수 필드 확인
                required_fields = [
                    "question", "intent", "needs_web", "plan",
                    "passages", "refined", "draft_answer", "citations",
                    "warnings", "trace", "web_results"
                ]
                
                missing_fields = [field for field in required_fields if field not in result]
                
                if not missing_fields:
                    print("✅ 하위 호환성 테스트 통과 - 모든 필수 필드 존재")
                else:
                    print(f"❌ 하위 호환성 테스트 실패 - 누락된 필드: {missing_fields}")
                    return False
            else:
                print(f"❌ 하위 호환성 테스트 실패: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 하위 호환성 테스트 오류: {e}")
            return False
        
        return True
    
    def cleanup(self):
        """테스트 정리"""
        if self.session_id:
            try:
                response = requests.delete(f"{self.base_url}/rag/session/{self.session_id}")
                if response.status_code == 200:
                    print(f"✅ 테스트 세션 정리 완료: {self.session_id}")
                else:
                    print(f"⚠️ 테스트 세션 정리 실패: {response.status_code}")
            except Exception as e:
                print(f"⚠️ 테스트 세션 정리 오류: {e}")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 멀티턴 대화 및 캐싱 기능 테스트 시작\n")
        
        tests = [
            ("기본 기능", self.test_basic_functionality),
            ("세션 관리", self.test_session_management),
            ("멀티턴 대화", self.test_multiturn_conversation),
            ("캐싱 성능", self.test_caching_performance),
            ("하위 호환성", self.test_backward_compatibility)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
                results.append((test_name, False))
        
        # 결과 요약
        print("\n" + "="*50)
        print("📊 테스트 결과 요약")
        print("="*50)
        
        passed = 0
        for test_name, result in results:
            status = "✅ 통과" if result else "❌ 실패"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과")
        
        # 정리
        self.cleanup()
        
        return passed == len(results)


def main():
    """메인 함수"""
    tester = MultiTurnTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        exit(0)
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        exit(1)


if __name__ == "__main__":
    main()
