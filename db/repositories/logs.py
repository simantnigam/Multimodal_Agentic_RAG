from sqlalchemy.ext.asyncio import AsyncSession

from db.models.logs import RetrievalLog, EvaluationLog


async def insert_retrieval_log(session: AsyncSession, log: RetrievalLog) -> RetrievalLog:
    session.add(log)
    await session.flush()
    return log


async def insert_evaluation_log(session: AsyncSession, log: EvaluationLog) -> EvaluationLog:
    session.add(log)
    await session.flush()
    return log
