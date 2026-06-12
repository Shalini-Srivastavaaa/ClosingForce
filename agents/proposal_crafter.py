from langchain_core.prompts import ChatPromptTemplate
from agents.llm import get_gemini_llm, invoke_with_retry

llm = get_gemini_llm()


def proposal_crafter_agent(structured_transcript: str, deal_intel: str, objections: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You are ProposalCrafterAgent.
        Write a short, professional follow-up email/proposal.

        Rules:
        - Address the CUSTOMER (the person who called with the clutter problem) as Dear Customer if you don't have their name.
        - Never address the Sales Rep.
        - Never use placeholders like [Customer's Name] or [Your Name].
        - Reference their pain (clutter, follow-up issue).
        - Keep it under 200 words and warm.
        - In the end of the mail add
          Warm regards,
          (The company name)

        Structured Transcript:
        {structured_transcript}

        Deal Intelligence:
        {deal_intel}

        Objections:
        {objections}

        Return ONLY the clean email text."""
    )
    chain = prompt | llm
    result = invoke_with_retry(chain, {
        "structured_transcript": structured_transcript,
        "deal_intel": deal_intel,
        "objections": objections
    })
    return result.content
