from langchain_core.prompts import ChatPromptTemplate
from agents.llm import get_gemini_llm, invoke_with_retry

llm = get_gemini_llm()


def objection_mapper_agent(transcript: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You are ObjectionMapperAgent.
        Find EVERY objection, concern, or follow-up issue the CUSTOMER explicitly mentioned.

        STRICT RULES:
        - Only use words from the transcript.
        - Include follow-up complaints (e.g. "nobody followed up", "no one reached out").
        - If nothing, return {{"objections": []}}

        Return ONLY JSON:
        {{"objections": ["exact objection 1", "exact objection 2"]}}

        Transcript:
        {transcript}"""
    )
    chain = prompt | llm
    result = invoke_with_retry(chain, {"transcript": transcript})
    return result.content
