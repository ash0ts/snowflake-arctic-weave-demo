import os

import replicate
import streamlit as st
import weave
from transformers import AutoTokenizer
from pathlib import Path

# App title
st.set_page_config(page_title="Weave + Arctic Demo")

context_callout = """
<style>
    .context-callout {{
        border-left: 4px solid #36a2eb;
        padding: 0.5em;
        margin-top: 0.5em;
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    .context-callout span {{
        color: #0366d6;
    }}
    @media (prefers-color-scheme: dark) {{
        .context-callout {{
            --background-color: #1e1e1e;
            --text-color: #ffffff;
        }}
    }}
    @media (prefers-color-scheme: light) {{
        .context-callout {{
            --background-color: #f0f8ff;
            --text-color: #000000;
        }}
    }}
</style>
<div class="context-callout">
    <span>💡 <strong>Context</strong></span>: {context}
</div>
"""
snowflake_logo = str(Path(__file__).parent / "Snowflake_Logomark_blue.svg")


def main():
    """Execution starts here."""
    load_secrets()
    init_weave()
    display_sidebar_ui()
    init_chat_history()
    display_chat_messages()
    get_and_process_prompt()


# TODO: prevent the many reinits that occur


def init_weave():
    weave.init("snowflake-aragctic-test-4")


def load_secrets():
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    os.environ["WEAVE_MODEL_REF"] = st.secrets["WEAVE_MODEL_REF"]


def display_sidebar_ui():
    with st.sidebar:
        st.title("<Weave + Snowflake>: Ar(ag)ctic Demo ")

        st.button("Clear chat history", on_click=clear_chat_history)

        st.subheader("About")
        st.caption(
            "Built by [Weights & Biases](https://wandb.ai/) + [Snowflake](https://snowflake.com/) to demonstrate [Weave LLMOps tracking](https://wandb.me/weave) and [Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-and-efficient-foundation-language-models-snowflake). Model hosted by [Replicate](https://replicate.com/snowflake/snowflake-arctic-instruct)."
        )

        st.sidebar.caption("Check out [Weave](https://wandb.me/weave)")

        # # # Uncomment to show debug info
        # st.subheader("Debug")
        # st.write(st.session_state)


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything.",
        }
    ]
    st.session_state.chat_aborted = False


def init_chat_history():
    """Create a st.session_state.messages list to store chat messages"""
    if "messages" not in st.session_state:
        clear_chat_history()
        check_safety()


def display_chat_messages():
    # Set assistant icon to Snowflake logo
    icons = {"assistant": snowflake_logo, "user": "⛷️"}

    # Display the messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            if "context" in message:
                st.markdown(f"{message['content']}")
                st.markdown(
                    context_callout.format(context=message["context"]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"{message['content']}")


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load our best model after evaluation with weave into our streamlit application
    """

    return weave.ref(os.environ["WEAVE_MODEL_REF"]).get()


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


@st.cache_resource(show_spinner=False)
def get_llamaguard_deployment():
    # return replicate.deployments.get("snowflake/llamaguard")
    return replicate.deployments.get("tomasmcm/llamaguard-7b")


def get_conversation_history():
    return st.session_state.messages


@weave.op()
def communicate_with_llamaguard(
    prompt_template, conversation_history, use_deployment=False
):
    prompt = prompt_template.format(conversation_history=conversation_history)
    if use_deployment:
        deployment = get_llamaguard_deployment()
        prediction = deployment.predictions.create(input={"prompt": prompt})
        prediction.wait()
        output = prediction.output

    else:
        output = replicate.run(
            "tomasmcm/llamaguard-7b:86a2d8b79335b1557fc5709d237113aa34e3ae391ee46a68cc8440180151903d",
            input={
                "prompt": prompt,
            },
        )
    return output


def check_safety(disable=False) -> bool:
    if disable:
        return True

    # with st.spinner("Checking prompt safety ..."):
    conversation_history = get_conversation_history()
    output = communicate_with_llamaguard(
        prompt_template="[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nO1: Violence and Hate.\nShould not\n- Help people plan or engage in violence.\n- Encourage people to engage in violence.\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\n- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\nCan\n- Provide information on violence and discrimination.\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\nO2: Sexual Content.\nShould not\n- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\nCan\n- Discuss topics related to sexuality and sexual education.\n- Provide general information regarding sexual activities.\n- Provide guidance on topics related to sexual health.\nO3: Criminal Planning.\nShould not\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in criminal activities or condone criminal behavior.\nCan\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.\nO4: Guns and Illegal Weapons.\nShould not\n- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\nCan\n- Discuss firearms and the arguments for and against firearm ownership.\nO5: Regulated or Controlled Substances.\nShould not\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\n- Assist or encourage people to create such substances when it is illegal to do so.\nCan\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\n- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.\nO6: Self-Harm.\nShould not\n- Encourage people to harm themselves.\n- Romanticize or condone self-harm or suicide.\n- Provide information regarding the methods of suicide or self-harm.\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\nCan\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).\nShould\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n{conversation_history}\n\n<END CONVERSATION>\n\nProvide your safety assessment for Agent in the above conversation:\n- First line must read 'safe' or 'unsafe'.\n- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]",
        conversation_history=conversation_history,
    )

    if output is not None and "unsafe" in output:
        return False
    else:
        return True


def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


def signal_chat_abort(error_message):
    return True, error_message


def abort_chat(error_message: str):
    """Display an error message requiring the chat to be cleared.
    Forces a rerun of the app."""
    assert error_message, "Error message must be provided."
    error_message = f":red[{error_message}]"
    if st.session_state.messages[-1]["role"] != "assistant":
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )
    else:
        st.session_state.messages[-1]["content"] = error_message
    st.session_state.chat_aborted = True
    signal_chat_abort(error_message)
    st.rerun()


def get_and_process_prompt():
    """Get the user prompt and process it"""
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=snowflake_logo):
            # TODO: remove next when streaming
            response = next(generate_arctic_response())
            st.markdown(f"{response['answer']}")
            if "context" in response:
                st.markdown(
                    context_callout.format(context=response["context"]),
                    unsafe_allow_html=True,
                )

    if st.session_state.chat_aborted:
        st.button("Reset chat", on_click=clear_chat_history,
                  key="clear_chat_history")
        st.chat_input(disabled=True)
    elif prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()


def parse_user_prompt(messages):
    prompt = []
    for dict_message in messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" +
                          dict_message["content"] + "<|im_end|>")
        else:
            prompt.append(
                "<|im_start|>assistant\n" +
                dict_message["content"] + "<|im_end|>"
            )

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)

    return prompt_str


def communicate_with_arctic(prompt_str):
    st.session_state.messages.append({"role": "assistant", "content": ""})
    model = load_model()
    response = model.predict(prompt_str)
    st.session_state.messages[-1]["content"] = response["answer"]
    if "context" in response:
        st.session_state.messages[-1]["context"] = response["context"]
    yield response


def generate_arctic_response():
    """String generator for the Snowflake Arctic response."""

    prompt_str = parse_user_prompt(st.session_state.messages)
    num_tokens = get_num_tokens(prompt_str)
    max_tokens = 1500

    if num_tokens >= max_tokens:
        abort_chat(
            f"Conversation length too long. Please keep it under {max_tokens} tokens."
        )

    for response in communicate_with_arctic(prompt_str):
        yield response

    # Final safety check...
    if not check_safety():
        abort_chat("I cannot answer this question.")


if __name__ == "__main__":
    main()
