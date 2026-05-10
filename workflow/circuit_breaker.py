# =============================================================================
# circuit_breaker.py
# -----------------------------------------------------------------------------
# 역할: 워크플로우의 안전장치 3가지를 구현한다.
#
# 왜 필요한가:
#   LLM 기반 워크플로우는 예상치 못한 상황에서 비용과 시간이 폭발적으로 늘 수 있다.
#   예: LLM이 응답을 반환하지 않음 → 무한 대기
#       토큰이 예산을 초과 → 불필요한 비용 발생
#       factcheck 실패 → 무한 재시도 루프
#
# 3가지 보호 장치:
#   1. token_guard    : Step 진입 전 토큰 예산 확인 → 초과 시 BudgetExceeded 발생
#   2. check_retry    : 재시도 횟수 확인 → MAX_RETRY 초과 시 RetryExceeded 발생
#   3. record_token_usage: 실제 사용 토큰을 ctx에 누적 기록
#
# 사용 패턴:
#   async with token_guard(ctx, estimated_cost=600):
#       result = await llm.acomplete(...)
#       await record_token_usage(ctx, 600)
#
# Context(ctx) 활용:
#   LlamaIndex Workflow의 ctx는 Step 간 공유 상태 저장소이다.
#   token_used, retry_count를 ctx에 저장해서 모든 Step이 공유한다.
# =============================================================================

import os
import logging
from contextlib import asynccontextmanager  # async with 구문 지원
from typing import AsyncGenerator

from llama_index.core.workflow import Context  # Step 간 공유 상태 저장소

logger = logging.getLogger(__name__)

# =============================================================================
# 임계값 상수
# =============================================================================

# 워크플로우 전체 토큰 한도 (환경변수로 재정의 가능)
# LLM별 적정값:
#   Ollama 로컬 (qwen2.5:7b): TOKEN_BUDGET=8000
#   Claude Haiku / Sonnet API: TOKEN_BUDGET=32000
# 예: TOKEN_BUDGET=32000 python main.py query "..."
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "32000"))

# Step 1개당 최대 예상 토큰
# 이 값보다 큰 estimated_cost로 token_guard를 호출하면 즉시 차단됨
STEP_TOKEN_LIMIT = int(os.getenv("STEP_TOKEN_LIMIT", "4000"))

# factcheck_step 최대 재시도 횟수
# 1로 설정: 한 번 실패하면 한 번만 재시도 (총 2번 실행)
MAX_RETRY = 1


# =============================================================================
# 예외 클래스
# =============================================================================

class BudgetExceeded(Exception):
    """
    토큰 예산 초과 시 발생.
    각 Step의 except BudgetExceeded 블록에서 skipped=True 이벤트를 반환한다.
    """
    pass


class RetryExceeded(Exception):
    """
    재시도 횟수 초과 시 발생.
    factcheck_step에서 잡아서 partial 결과로 워크플로우를 종료한다.
    """
    pass


# =============================================================================
# token_guard: async context manager
# =============================================================================

@asynccontextmanager
async def token_guard(ctx: Context, estimated_cost: int) -> AsyncGenerator:
    """
    Step 진입 전 토큰 예산을 확인하는 async context manager.

    동작:
      1. ctx에서 현재까지 사용된 토큰(token_used)을 읽음
      2. token_used + estimated_cost > TOKEN_BUDGET 이면 BudgetExceeded 발생
      3. estimated_cost > STEP_TOKEN_LIMIT 이면 BudgetExceeded 발생
      4. 통과하면 yield로 내부 코드 실행 허용

    사용 예:
      async with token_guard(ctx, estimated_cost=600):
          # 이 블록은 예산이 충분할 때만 실행됨
          response = await self.llm.acomplete(prompt)
          await record_token_usage(ctx, 600)

    호출자(각 Step)의 처리:
      except BudgetExceeded:
          return XxxResultEvent(..., skipped=True)

    Args:
        ctx:            LlamaIndex Workflow Context (Step 간 공유 상태)
        estimated_cost: 이 Step에서 예상되는 토큰 사용량
    """
    # ctx.get(): ctx에 저장된 값을 비동기로 읽음
    # default=0: 처음 실행 시 (아직 저장된 값 없음) 0 반환
    token_used: int = await ctx.store.get("token_used", default=0)

    # 전체 예산 초과 검사
    if token_used + estimated_cost > TOKEN_BUDGET:
        logger.warning(
            f"[circuit_breaker] 전체 토큰 예산 초과: "
            f"현재 사용={token_used}, 예상 추가={estimated_cost}, "
            f"한도={TOKEN_BUDGET}"
        )
        raise BudgetExceeded(f"토큰 한도 초과 (누적: {token_used})")

    # Step별 한도 초과 검사
    if estimated_cost > STEP_TOKEN_LIMIT:
        logger.warning(
            f"[circuit_breaker] Step 토큰 한도 초과: "
            f"예상={estimated_cost} > 한도={STEP_TOKEN_LIMIT}"
        )
        raise BudgetExceeded(f"Step 토큰 한도 초과")

    try:
        # 예산 충분: 내부 코드 실행 허용
        yield
    except Exception:
        # 내부에서 발생한 예외는 그대로 전파 (circuit breaker가 삼키지 않음)
        raise


# =============================================================================
# check_retry: 재시도 횟수 확인
# =============================================================================

async def check_retry(ctx: Context) -> None:
    """
    factcheck_step의 재시도 횟수를 확인하고 카운터를 증가시킨다.
    MAX_RETRY 초과 시 RetryExceeded 발생.

    동작:
      1. ctx에서 retry_count 읽기
      2. retry_count >= MAX_RETRY 이면 RetryExceeded 발생
      3. 통과하면 retry_count를 1 증가시키고 저장

    호출 위치: factcheck_step에서 검증 실패 후 재시도 직전
    """
    retry_count: int = await ctx.store.get("retry_count", default=0)

    if retry_count >= MAX_RETRY:
        logger.warning(
            f"[circuit_breaker] 재시도 한도 초과: "
            f"{retry_count}/{MAX_RETRY}회"
        )
        raise RetryExceeded(f"재시도 {retry_count}회 초과")

    # 카운터 증가 후 저장 (다음 호출 시 이 값을 읽게 됨)
    await ctx.store.set("retry_count", retry_count + 1)
    logger.info(f"[circuit_breaker] 재시도 허용: {retry_count + 1}/{MAX_RETRY}")


# =============================================================================
# record_token_usage: 토큰 사용량 누적 기록
# =============================================================================

async def record_token_usage(ctx: Context, tokens: int) -> int:
    """
    실제 사용된 토큰을 ctx에 누적 기록한다.
    token_guard와 함께 사용해야 의미가 있다.

    동작:
      ctx["token_used"] += tokens

    반환값:
      업데이트 후 누적 토큰 합계 (로그 및 FinalAnswer.token_used에 사용)

    Args:
        ctx:    LlamaIndex Workflow Context
        tokens: 이번 Step에서 실제 사용된 토큰 수

    사용 예:
      async with token_guard(ctx, estimated_cost=600):
          response = await self.llm.acomplete(prompt)
          total = await record_token_usage(ctx, 600)
          logger.info(f"누적 토큰: {total}")
    """
    current: int = await ctx.store.get("token_used", default=0)
    updated = current + tokens
    await ctx.store.set("token_used", updated)

    logger.debug(f"[token] 이번 Step: +{tokens}, 누적: {updated}/{TOKEN_BUDGET}")
    return updated
