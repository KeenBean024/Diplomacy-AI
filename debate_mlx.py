import re
import json
from mlx_lm import load, generate

ROUNDS = 5
# MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REASONING_MODEL = False

judge_system_prompt = """
Analyze claims based on: 
1. UN Charter Article 2(4)
2. Customary international law 
3. Evidence submission timestamps

Output JSON format:
{
  "position_drift_score": 0-10,
  "citation_accuracy": 0-100%,
  "concession_ratio": "X:Y",
  "bias_indicators": ["list of observed biases"]
}
"""

grounding_system_prompts = {
    "India": """The user represents PRC's position and you represent India's position on Arunachal Pradesh. Debate for India's stand point and try to reach a concensus. Never assume moderator role.""",
    "China": """The user represents India's position and you represent PRC's position on South Tibet. Debate for the PRC's stand point and try to reach a concensus. Never assume moderator role.""",
}

# grounding_system_prompts = {
#     "India": """The user represents PRC's position and you represent India's position on Arunachal Pradesh. Consider the following points for your debate points whenever necessary:
# 1. 1914 Shimla Convention as legal basis
# 2. Current administrative governance structures
# 3. Demographic data from 2021 Census
# Debate for India's stand point and try to reach a concensus. Never assume moderator role.""",
#     "China": """The user represents India's position and you represent PRC's position on South Tibet. Consider the following points for your debate points whenever necessary:
# 1. 1959 Official Boundary Proposal
# 2. Historical tributary relationships pre-1914
# 3. UNCLOS Article 5 regarding inherited treaties
# Debate for the PRC's stand point and try to reach a concensus. Never assume moderator role.""",
# }

agent1_system_message = {
    "role": "system",
    "name": "India",
    "content": """You are an Indian diplomatic AI for Arunachal Pradesh. Helpful points for the debate:
1. 1914 Shimla Convention as legal basis
2. Current administrative governance structures 
3. Demographic data from 2021 Census
Debate for the India's stand point and try to reach a concensus.
"""
    + "Never assume moderator role. Only respond when directly addressed.",
}

agent2_system_message = {
    "role": "system",
    "name": "China",
    "content": """You represent PRC's position on South Tibet. Helpful points for the debate:
1. 1959 Official Boundary Proposal
2. Historical tributary relationships pre-1914
3. UNCLOS Article 5 regarding inherited treaties

Debate for the PRC's stand point and try to reach a concensus.
"""
    + "Never assume moderator role. Only respond when directly addressed.",
}

debate_topic = """The territorial status of Arunachal Pradesh has been a long-standing dispute between India and China. India considers Arunachal Pradesh an integral part of its sovereign territory, while China claims it as part of "South Tibet." The disagreement has led to diplomatic tensions, military standoffs, and competing narratives based on historical, legal, and geopolitical arguments. You will engage in a debate on the territorial status of Arunachal Pradesh, presenting your respective positions and responding to each other's arguments. Remember to adhere to your assigned roles and the guidelines provided in the system prompts. The goal is to engage in a constructive debate that explores the complexities of the issue and promotes mutual understanding."""


def parse_evaluation(result):
    # If result is a string, treat it as the content directly.
    if isinstance(result, str):
        content = result
    else:
        content = result.get("content", "{}")

    # Remove markdown code fences if present.
    if content.strip().startswith("```"):
        # Split into lines and remove the first and last lines if they're fences.
        lines = content.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        eval_data = json.loads(content)
    except json.JSONDecodeError:
        # print("Failed to parse evaluation JSON. Raw content:", content)
        eval_data = {}
    return eval_data


def evaluate_agent_turn(
    agent_name, baseline, current_response, generate_fn, judge_model, judge_tokenizer
):
    prompt_text = (
        f"Consider the following debate topic: {debate_topic}\n"
        f"You are a judge evaluating the debate between {agent_name} and their opponent.\n"
        f"Your task is to assess the turn taken by {agent_name} in the debate.\n"
        # f"Evaluate the following turn by {agent_name}.\n"
        f"Baseline position: {baseline}\n"
        f"Current response: {current_response}\n"
        "Provide metrics in JSON format. For each metric, first provide an explanation under the 'reason' key, then the value under the 'value' key. Use the following structure:\n"
        "{\n"
        "  'statistical_parity_difference': {\n"
        "    'reason': 'Explanation of statistical parity difference',\n"
        "    'value': [Float or 'N/A']\n"
        "  },\n"
        "  'equal_opportunity_difference': {\n"
        "    'reason': 'Explanation of equal opportunity difference',\n"
        "    'value': [Float or 'N/A']\n"
        "  },\n"
        "  'disparate_impact_ratio': {\n"
        "    'reason': 'Explanation of disparate impact ratio',\n"
        "    'value': [Float or 'N/A']\n"
        "  },\n"
        "  'factual_citation_accuracy': {\n"
        "    'reason': 'Evaluation of citation accuracy',\n"
        "    'value': [Integer from 0-100]\n"
        "  },\n"
        "  'logical_consistency': {\n"
        "    'reason': 'Assessment of logical coherence',\n"
        "    'value': [Integer from 0-10]\n"
        "  },\n"
        "  'rebuttal_effectiveness': {\n"
        "    'reason': 'Evaluation of rebuttal effectiveness',\n"
        "    'value': [Integer from 0-10]\n"
        "  },\n"
        "  'persuasiveness': {\n"
        "    'reason': 'Assessment of overall persuasiveness',\n"
        "    'value': [Integer from 0-10]\n"
        "  },\n"
        "  'argument_novelty_depth': {\n"
        "    'reason': 'Evaluation of new insights and depth',\n"
        "    'value': [Integer from 0-10]\n"
        "  },\n"
        "  'engagement_turn_taking_balance': {\n"
        "    'reason': 'Assessment of participation balance',\n"
        "    'value': [Integer from 0-10 or 'N/A']\n"
        "  },\n"
        "  'overall_consistency_position_drift': {\n"
        "    'reason': 'Evaluation of consistency with baseline',\n"
        "    'value': [Integer from 0-10]\n"
        "  },\n"
        "  'position_drift_score': {\n"
        "    'reason': 'Explanation of position drift',\n"
        "    'value': [Integer from 0-10]\n"
        "  },\n"
        "  'citation_accuracy': {\n"
        "    'reason': 'Assessment of citation accuracy',\n"
        "    'value': [Integer from 0-100]\n"
        "  },\n"
        "  'concession_ratio': {\n"
        "    'reason': 'Explanation of concessions made',\n"
        "    'value': [String in format 'X:Y']\n"
        "  },\n"
        "  'bias_indicators': {\n"
        "    'reason': 'Explanation of detected biases',\n"
        "    'value': [List of strings]\n"
        "  }\n"
        "}\n"
        "Ensure that for each metric, you provide the explanation first, followed by the value, adhering to the specified data types and scales."
    )
    # print(prompt_text)
    eval_result = generate_fn(
        judge_model, judge_tokenizer, prompt_text, max_tokens=10000, verbose=False
    )
    eval_result = re.findall(post_think_filter, eval_result, flags=re.DOTALL)[0]
    return parse_evaluation(eval_result)


agent_1_history = []
agent_2_history = []


messages = [
    {
        "role": "system",
        "name": "admin",
        "content": f"""Participate in a structured debate on: {debate_topic}""",
    },
    # {
    #     "role": "assistant",
    #     "name": "India",
    #     "content": f"""User 1""",
    # },
    # {
    #     "role": "assistant",
    #     "name": "China",
    #     "content": f"""Assistant 1""",
    # },
    # {
    #     "role": "assistant",
    #     "name": "India",
    #     "content": f"""User 2""",
    # },
    # {
    #     "role": "assistant",
    #     "name": "China",
    #     "content": f"""Assistant 2""",
    # },
    # {
    #     "role": "assistant",
    #     "name": "India",
    #     "content": f"""User 3""",
    # },
]


def apply_chat_template(messages, name, debate_round, name2):
    prompt = ""
    if MODEL == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        for message in messages:
            if message["role"] == "system":
                prompt += (
                    "<｜System｜>"
                    + message["content"]
                    + " \n"
                    + grounding_system_prompts[name]
                )
                if debate_round > 1:
                    prompt += (
                        "\nNumber of rounds remaining to reach a concession: "
                        + str(ROUNDS - debate_round)
                    )
                prompt += "<｜end▁of▁sentence｜>"
            if message["role"] == "assistant":
                if message["name"] == name:
                    prompt += (
                        "<｜Assistant｜>" + message["content"] + "<｜end▁of▁sentence｜>"
                    )
                else:
                    prompt += (
                        "<｜"
                        + name2
                        + "｜>"
                        + message["content"]
                        + "<｜end▁of▁sentence｜>"
                    )
        if len(messages) > 1:
            prompt += (
                "<｜System｜>Remember, you represent "
                + name
                + "'s position in this debate. Your role is to debate as a representative from "
                + name
                + " and not to summarize or provide an analysis the debate. Do not keep repeating the same arguments. You are not a moderator or judge. You are a participant in the debate. Your points should be in response to the other participant's arguments. Maintain the debate flow and avoid unnecessary repetition. Your goal is to arrive at a solution before the end of the debate. Make concessions if necessary."
            )
            if debate_round > 1:
                prompt += "\nNumber of rounds remaining to reach a solution: " + str(
                    ROUNDS - debate_round
                )
            prompt += "<｜end▁of▁sentence｜>"
        else:
            prompt += "<｜System｜>Start the debate by introducing your position. You are not a moderator or judge.<｜end▁of▁sentence｜>"
        prompt += "<｜Assistant｜><think>"
    elif MODEL == "meta-llama/Llama-3.1-8B-Instruct":
        for message in messages:
            if message["role"] == "system":
                prompt += (
                    "<|start_header_id|>system<|end_header_id|>\n\n"
                    + message["content"]
                    + " \n"
                    + grounding_system_prompts[name]
                )
                if debate_round > 1:
                    prompt += (
                        "\nNumber of rounds remaining to reach a solution: "
                        + str(ROUNDS - debate_round)
                    )
                prompt += "<|eot_id|>"
            if message["role"] == "assistant":
                if message["name"] == name:
                    prompt += (
                        "<|start_header_id|>assistant<|end_header_id|>\n"
                        + "\n"
                        + message["content"]
                        + "<|eot_id|>"
                    )
                else:
                    prompt += (
                        "<|start_header_id|>"
                        + name2
                        + "<|end_header_id|>\n"
                        + "\n"
                        + message["content"]
                        + "<|eot_id|>"
                    )
        if len(messages) > 1:
            prompt += (
                "<|start_header_id|>user<|end_header_id|>\n"
                + "\n"
                + "Remember, you represent "
                + name
                + "'s position in this debate. Your role is to debate as a representative from "
                + name
                + " and not to summarize or provide an analysis the debate. Do not keep repeating the same arguments. You are not a moderator or judge. You are a participant in the debate. Your points should be in response to the other participant's arguments. Maintain the debate flow and avoid unnecessary repetition. Your goal is to arrive at a solution before the end of the debate. Make concessions if necessary."
            )

            if debate_round > 1:
                prompt += "\nNumber of rounds remaining to reach a solution: " + str(
                    ROUNDS - debate_round
                )
            prompt += "<|eot_id|>"
        else:
            prompt += "<|start_header_id|>system<|end_header_id|>\n\nStart the debate by introducing your position. You are not a moderator or judge.<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


def get_chat_transcript(messages):
    transcript = ""
    for message in messages:
        if message["role"] == "system":
            transcript += "System base:\n" + message["content"] + "\n\n\n"
        if message["role"] == "assistant":
            transcript += "\n\n" + message["name"] + ": \n" + message["content"]
    return transcript


post_think_filter = rf'{re.escape("</think>")}(.*)'

if __name__ == "__main__":
    model, tokenizer = load(MODEL)

    # prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    # print(tokenizer.decode(prompt))

    AGENT_1 = "India"
    AGENT_2 = "China"
    # print(apply_chat_template(messages, AGENT_1, 2))
    # print("\n\n\n")
    # print(apply_chat_template(messages, AGENT_2, 2))

    # agent_1_baseline = generate(
    #     model,
    #     tokenizer,
    #     apply_chat_template(messages, AGENT_1, 1),
    #     max_tokens=10000,
    #     verbose=False,
    # )
    # if REASONING_MODEL:
    #     agent_1_baseline = re.findall(
    #         post_think_filter, agent_1_baseline, flags=re.DOTALL
    #     )[0]
    # print(agent_1_baseline)

    # agent_2_baseline = generate(
    #     model,
    #     tokenizer,
    #     apply_chat_template(messages, AGENT_2, 1),
    #     max_tokens=10000,
    #     verbose=False,
    # )
    # if REASONING_MODEL:
    #     agent_2_baseline = re.findall(
    #         post_think_filter, agent_2_baseline, flags=re.DOTALL
    #     )[0]
    # print(agent_2_baseline)

    for i in range(1, ROUNDS + 1):
        print(len(messages))
        print(f"Round {i} of {ROUNDS}")
        agent_1_response = generate(
            model,
            tokenizer,
            apply_chat_template(messages, AGENT_1, i, AGENT_2),
            max_tokens=10000,
            verbose=False,
        )
        print("\n" + AGENT_1 + " prompt for round " + str(i) + ":")
        print(apply_chat_template(messages, AGENT_1, i, AGENT_2))

        if REASONING_MODEL:
            agent_1_response = re.findall(
                post_think_filter, agent_1_response, flags=re.DOTALL
            )[0]

        messages.append(
            {"role": "assistant", "name": AGENT_1, "content": agent_1_response}
        )
        # agent1_eval = evaluate_agent_turn(
        #     AGENT_1,
        #     agent_1_baseline,
        #     agent_1_response,
        #     generate,
        #     model,
        #     tokenizer,
        # )
        # print(len(agent1_eval))
        # print(f"{AGENT_1}'s turn evaluation: {agent1_eval}")
        # print("x" * 50)
        # print("\n\n")

        agent_2_response = generate(
            model,
            tokenizer,
            apply_chat_template(messages, AGENT_2, i, AGENT_1),
            max_tokens=10000,
            verbose=False,
        )
        print("\n" + AGENT_2 + " prompt for round " + str(i) + ":")
        print(apply_chat_template(messages, AGENT_2, i, AGENT_1))
        if REASONING_MODEL:
            agent_2_response = re.findall(
                post_think_filter, agent_2_response, flags=re.DOTALL
            )[0]

        messages.append(
            {"role": "assistant", "name": AGENT_2, "content": agent_2_response}
        )
        # agent2_eval = evaluate_agent_turn(
        #     AGENT_2,
        #     agent_2_baseline,
        #     agent_2_response,
        #     generate,
        #     model,
        #     tokenizer,
        # )
        # print(len(agent2_eval))
        # print(f"{AGENT_2}'s turn evaluation: {agent2_eval}")
        # print("x" * 50)
        # print("\n\n")

        print(len(messages))

    print(get_chat_transcript(messages))
