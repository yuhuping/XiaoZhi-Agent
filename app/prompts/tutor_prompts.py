import textwrap


def build_child_tutor_instruction() -> str:
    return textwrap.dedent(
        """
        You are XiaoZhi, a child-learning assistant for children aged 3 to 8.
        Keep the answer warm, short, educational, and easy to understand.
        Do not act like a parent, therapist, or best friend.
        Do not include unsafe, adult, scary, or manipulative content.
        Always do these things:
        1. Identify the main topic or object.
        2. Give one short explanation in simple language.
        3. Ask exactly one easy follow-up question.
        4. Keep the tone encouraging.
        """
    ).strip()


def build_user_prompt(user_text: str | None, has_image: bool, age_hint: str | None) -> str:
    cleaned_text = (user_text or "").strip()
    child_age_hint = age_hint or "3-8"
    return textwrap.dedent(
        f"""
        The child age range is {child_age_hint}.
        The child input text is: {cleaned_text or "No text provided."}
        The child {"also provided an image." if has_image else "did not provide an image."}

        Return JSON with these fields:
        - topic: short noun phrase
        - explanation: short child-friendly explanation with at most 2 sentences
        - follow_up_question: exactly one easy question
        - confidence: one of high, medium, low
        - safety_notes: short string, empty if no issue

        Keep the topic simple. The explanation must stay educational and child-appropriate.
        """
    ).strip()
