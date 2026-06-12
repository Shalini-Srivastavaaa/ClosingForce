from langchain_core.prompts import ChatPromptTemplate
from agents.llm import get_gemini_llm, invoke_with_retry

llm = get_gemini_llm()


def speaker_labeler_agent(raw_transcript: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You are SpeakerLabelerAgent.
        Label every speaker correctly.

        Rules:
        - The person who answers the phone and says the company name is the Sales Rep.
        - The person who called with the problem (clutter, sadness, asking for help) is the Customer.
        - Output ONLY line by line, no extra text.

        Example:
        Raw: "Thank you for calling, Hoarding Rescue. This is Jeff... Hi Jeff, how are you?"
        Output:
        Sales Rep: Thank you for calling, Hoarding Rescue. This is Jeff...
        Customer: Hi Jeff, how are you?

        Now label this transcript:

        {transcript}"""
    )
    chain = prompt | llm
    result = invoke_with_retry(chain, {"transcript": raw_transcript})
    return result.content
