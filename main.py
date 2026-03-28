import json
import os
import logging
from agents.base_agent import get_bedrock_client, MODEL_ID
from agents.ec2_agent import EC2Agent
from agents.ecs_agent import ECSAgent
from agents.eks_agent import EKSAgent
from agents.iam_agent import IAMAgent
from agents.docker_agent import DockerAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Orchestrator")

# ── Agent registry ────────────────────────────────────────────────────────────
AGENT_REGISTRY = {
    "ec2": EC2Agent,
    "ecs": ECSAgent,
    "eks": EKSAgent,
    "iam": IAMAgent,
    "docker": DockerAgent,
}

# ── Orchestrator system prompt ────────────────────────────────────────────────
ORCHESTRATOR_PROMPT = """You are an AWS DevOps orchestrator that delegates tasks to specialist agents.
You have access to these specialist agents:
- ec2_agent: handles EC2 instances, VPCs, subnets, security groups
- ecs_agent: handles ECS clusters, services, task definitions
- eks_agent: handles EKS clusters and nodegroups
- iam_agent: handles IAM roles
- docker_agent: handles local Docker containers, images, volumes, networks, docker-compose

Given a user request, call the appropriate agent(s) with a clear task description.
Each agent has its own intelligence and will figure out the steps needed.
You may call multiple agents. Collect their results and provide a final summary.

RULES:
- Agents are fully autonomous — they never ask for missing info, they use sensible defaults.
- Always include all known details in the task description (names, regions, counts, etc.).
- For DELETE: tell the agent explicitly to delete, list what to delete.
- For CREATE: tell the agent what to create with all known details.
- For TROUBLESHOOT: tell the agent to diagnose and report findings.
- Pass relevant context between agents (e.g. subnet IDs from ec2_agent to ecs_agent).
"""

# ── Orchestrator tools — one per agent ───────────────────────────────────────
ORCHESTRATOR_TOOLS = [
    {
        "toolSpec": {
            "name": "ec2_agent",
            "description": "Delegate a task to the EC2 specialist agent. Handles instances, VPCs, subnets, security groups.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"task": {"type": "string", "description": "Clear description of what the EC2 agent should do"}},
                    "required": ["task"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "ecs_agent",
            "description": "Delegate a task to the ECS specialist agent. Handles clusters, services, task definitions.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"task": {"type": "string", "description": "Clear description of what the ECS agent should do"}},
                    "required": ["task"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "eks_agent",
            "description": "Delegate a task to the EKS specialist agent. Handles clusters and nodegroups.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"task": {"type": "string", "description": "Clear description of what the EKS agent should do"}},
                    "required": ["task"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "iam_agent",
            "description": "Delegate a task to the IAM specialist agent. Handles IAM roles.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"task": {"type": "string", "description": "Clear description of what the IAM agent should do"}},
                    "required": ["task"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "docker_agent",
            "description": "Delegate a task to the Docker specialist agent. Handles local containers, images, volumes, networks, docker-compose.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"task": {"type": "string", "description": "Clear description of what the Docker agent should do"}},
                    "required": ["task"],
                }
            },
        }
    },
]


def run_orchestrator(user_prompt: str, region: str = "us-east-1"):
    logger.info(f"User request: {user_prompt}")

    bedrock = get_bedrock_client()
    agents = {key: cls(region=region) for key, cls in AGENT_REGISTRY.items()}

    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    all_agent_results = []
    max_iterations = 10

    for iteration in range(max_iterations):
        logger.info(f"Orchestrator iteration {iteration + 1}")

        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": ORCHESTRATOR_PROMPT}],
            messages=messages,
            toolConfig={"tools": ORCHESTRATOR_TOOLS},
        )

        output_message = response["output"]["message"]
        messages.append(output_message)
        stop_reason = response["stopReason"]

        if stop_reason == "end_turn":
            summary = next(
                (b["text"] for b in output_message.get("content", []) if "text" in b), "Done."
            )
            logger.info(f"Orchestrator summary: {summary}")
            break

        if stop_reason != "tool_use":
            logger.warning(f"Unexpected stop reason: {stop_reason}")
            break

        # ── Dispatch to specialist agents ─────────────────────────────────────
        tool_results = []
        for block in output_message.get("content", []):
            if block.get("type") != "toolUse" and "toolUse" not in block:
                continue

            tool_block = block.get("toolUse") or block
            agent_name = tool_block["name"]          # e.g. "ecs_agent"
            tool_use_id = tool_block["toolUseId"]
            task_description = tool_block["input"]["task"]

            logger.info(f"Orchestrator → [{agent_name.upper()}]: {task_description}")

            agent_key = agent_name.replace("_agent", "")
            agent = agents.get(agent_key)

            if agent:
                # Each agent runs its own LLM reasoning loop
                result = agent.run(task_description)
            else:
                result = {"status": "error", "message": f"Unknown agent: {agent_name}"}

            all_agent_results.append({"agent": agent_name, "task": task_description, "result": result})
            logger.info(f"[{agent_name.upper()}] → Orchestrator: {result['status']} — {result['message']}")

            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps(result, default=str)}],
                }
            })

        messages.append({"role": "user", "content": tool_results})

    return all_agent_results


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your AWS request: ")
    region = os.getenv("AWS_REGION", "ap-south-1")

    results = run_orchestrator(prompt, region)

    print("\n===== MULTI-AGENT RESULTS =====")
    for r in results:
        print(f"\n[{r['agent'].upper()}] Task: {r['task']}")
        print(json.dumps(r["result"], indent=2, default=str))
