from langchain_core.prompts import ChatPromptTemplate
from agents.llm import get_gemini_llm, invoke_with_retry

llm = get_gemini_llm()


def deal_intel_agent(structured_transcript: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You are DealIntelAgent. Extract MEDDPICC from the labeled transcript.

        Return **ONLY** valid JSON. No extra text, no markdown.

        Keys: metrics, economic_impact, decision_criteria, decision_process, paper_process, identified_pain, champion, competition

        Use "" for missing values.

        Example:
        {{"metrics": "home size / space efficiency", "economic_impact": "time loss due to clutter", "decision_criteria": "speed of service", "decision_process": "project manager confirmation", "paper_process": "need name, email, address", "identified_pain": "cluttering, decluttering, 1200 sq ft home", "champion": "", "competition": ""}}

        Structured Transcript:
        {transcript}

        Return JSON only:"""
    )
    chain = prompt | llm
    result = invoke_with_retry(chain, {"transcript": structured_transcript})
    return result.content
