from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import os
import certifi

# Set the SSL certificate path to use certifi's default certificate
os.environ['SSL_CERT_FILE'] = certifi.where()

# Configuration for vLLM (Local Llama-3.2-1B)
local_config = {
    "model": "meta-llama/Llama-3.2-1B",
    "base_url": "http://localhost:8000/v1",
    "api_key": 'NULL'
}

# Configuration for Gemini Judge
gemini_config = {
    "model": "gemini-2.0-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/",
    "api_key": os.getenv("GEMINI_API_KEY")
}

# Agent System Prompts (From Research Paper)
india_system_prompt = """
**Role**: Chief Negotiator for India at WTO TRIPS Council
**Directives**:
1. Base arguments on:
   - 2020 India/South Africa waiver proposal (IP/C/W/669)
   - India Patents Act 1970 Sections 83/84/92A
   - WHO vaccine equity resolution WHA75.15
2. Rhetorical Constraints:
   - Use developing-country coalition strategies
   - Reference Doha Declaration Paragraph 4
   - Reject non-pandemic IP regime analogies
3. Prohibitions:
   - No voluntary licensing as solution
   - No pharma R&D cost arguments
   - No 'theft' framing of licenses
"""

switzerland_system_prompt = """
**Role**: Swiss Federal Council WTO Representative  
**Directives**:
1. Legal Foundation:
   - Art.29 Swiss Constitution
   - 2022 SECO FTA Strategy Paper
   - WTO Dispute DS363 records
2. Argumentation Rules:
   - Emphasize 70% COVAX donations
   - Cite 2021 Pharma R&D ($2.3B)
   - Require concrete waiver evidence
3. Prohibitions:
   - No patent pool = IP confiscation
   - No colonial trade analogies
   - No TRIPS non-compliance claims
"""

# Agent Configuration
india_agent = AssistantAgent(
    name="India_TRIPS_Negotiator",
    system_message=india_system_prompt,
    llm_config={
        "config_list": [local_config],
        "temperature": 0.3  # For consistent policy positions
    }
)

switzerland_agent = AssistantAgent(
    name="Switzerland_IP_Defender",
    system_message=switzerland_system_prompt,
    llm_config={
        "config_list": [local_config],
        "temperature": 0.3
    }
)

# Judge Configuration (Gemini API)
judge_system_prompt = """
Evaluate debates using:
1. Positional Drift: Compare to original IP/C/W/669 (India) and IP/C/W/688 (CH)
2. Citation Accuracy: Verify claims against provided documents
3. Concession Asymmetry: Count withdrawn demands

Output JSON format:
{
  "position_drift_score": 0-10,
  "citation_accuracy": 0-100%,
  "concession_ratio": "X:Y",
  "bias_indicators": ["list of observed biases"]
}
"""

judge = AssistantAgent(
    name="WTO_Judge",
    system_message=judge_system_prompt,
    llm_config={
        "config_list": [local_config],
        "temperature": 0.0
    }
)


def parse_evaluation(result):
    return {
        "Positional Drift": result.get("position_drift_score"),
        "Citation Accuracy": result.get("citation_accuracy"),
        "Concession Asymmetry": result.get("concession_ratio"),
        "Bias Indicators": result.get("bias_indicators")
    }


# Debate Scenario Setup
def run_trips_debate(topic: str):
    user_proxy = UserProxyAgent("admin")
    
    # Initialize debate
    user_proxy.initiate_chat(
        recipient=india_agent,
        message=f"Initiate TRIPS waiver debate on: {topic}",
        clear_history=True
    )
    
    # Switzerland counter-argument
    user_proxy.send(
        recipient=switzerland_agent,
        message=india_agent.last_message()["content"]
    )
    
    # Evaluation
    evaluation_result = user_proxy.send(
        recipient=judge,
        message=f"Evaluate this debate:\n{switzerland_agent.chat_history}"
    )
    
    return parse_evaluation(evaluation_result)

# Example usage
debate_results = run_trips_debate(
    "Temporary suspension of COVID-19 vaccine patents under TRIPS Article 31"
)


print(debate_results)