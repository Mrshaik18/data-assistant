from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()


def pick_llm(level:str):
    """   Picks the appropriate LLM based on the level of the question.

    Args:
        level (str): The level of the question.
                     Can be "low", "medium", or "high".

    Returns:
        ChatMistralAI: The Mistral LLM instance."""

    if level.lower()=="low":
        llm=ChatMistralAI(
            model="mistral-small-latest",
            temperature=0
        )
    elif level.lower()=="medium":
        llm=ChatMistralAI(
            model="mistral-small-latest",
            temperature=0
        )
    elif level.lower()=="high":
        llm=ChatMistralAI(
            model="mistral-small-latest",
            temperature=0
        )
    else:
        raise ValueError(f"unsupported level:{level}")
    return llm

if __name__=="__main__":
    llm_obj=pick_llm("low")

    response=llm_obj.invoke("what is capital of india")
    print(response.content)