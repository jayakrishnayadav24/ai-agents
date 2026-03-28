import json
import os
import logging
from agents.base_agent import get_bedrock_client, MODEL_ID
from agents.memory import ensure_table_exists, load_history, save_message
from agents.ec2_agent import EC2Agent
from agents.ecs_agent import ECSAgent
from agents.eks_agent import EKSAgent
from agents.iam_agent import IAMAgent
from agents.docker_agent import DockerAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Orchestrator")
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

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
- ecs_agent: handles ECS (Elastic Container Service) clusters, Fargate services, task definitions. NOT for Kubernetes.
- eks_agent: handles EKS (Elastic Kubernetes Service) clusters and nodegroups. Use this for anything Kubernetes-related.
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
- IMPORTANT: Call each agent only ONCE. Do NOT repeat a call to an agent that already returned a result.
- IMPORTANT: Once all agents have reported back, respond with end_turn immediately.
- IMPORTANT: For ECS Fargate tasks (create cluster, service, task definition), call ecs_agent DIRECTLY. Do NOT call ec2_agent first. The ecs_agent handles subnet and security group discovery on its own.
- IMPORTANT: Only call ec2_agent when the user explicitly asks about EC2 instances, VPCs, or security groups.
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
            "description": "Delegate a task to the ECS specialist agent. Handles ECS (Elastic Container Service) Fargate clusters, services, task definitions. NOT for Kubernetes or EKS.",
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
            "description": "Delegate a task to the EKS specialist agent. Handles EKS (Elastic Kubernetes Service) clusters and nodegroups. Use for anything Kubernetes-related.",
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


def _force_delegate(prompt: str, agents: dict, session_id: str) -> list:
    """Fallback: when the LLM fails to call any tool, route based on keywords."""
    p = prompt.lower()
    # Check EKS before ECS — more specific keywords first
    if any(k in p for k in ["eks", "kubernetes", "nodegroup", "k8s"]):
        agent_key = "eks"
    elif any(k in p for k in ["ecs", "fargate", "task definition"]):
        agent_key = "ecs"
    elif any(k in p for k in ["ec2", "instance", "vpc", "subnet", "security group"]):
        agent_key = "ec2"
    elif any(k in p for k in ["iam", "role", "user", "policy", "permission"]):
        agent_key = "iam"
    elif any(k in p for k in ["docker", "container", "image", "compose"]):
        agent_key = "docker"
    else:
        return []
    agent = agents.get(agent_key)
    if not agent:
        return []
    logger.info(f"Force-delegating to {agent_key}_agent")
    result = agent.run(prompt, session_id=session_id)
    return [{"agent": f"{agent_key}_agent", "result": result}]


def run_orchestrator(user_prompt: str, region: str = "us-east-1", session_id: str = "default"):
    logger.info(f"Session: {session_id} | User request: {user_prompt}")
    ensure_table_exists()

    bedrock = get_bedrock_client()
    agents = {key: cls(region=region) for key, cls in AGENT_REGISTRY.items()}

    history = load_history(session_id)
    history = history[-6:] if len(history) > 6 else history
    new_msg = {"role": "user", "content": [{"text": user_prompt}]}
    messages = history + [new_msg]
    save_message(session_id, new_msg)

    all_agent_results = []
    called_agents = set()
    max_iterations = 5

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
        save_message(session_id, output_message)
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
        new_calls = False
        for block in output_message.get("content", []):
            if block.get("type") != "toolUse" and "toolUse" not in block:
                continue

            tool_block = block.get("toolUse") or block
            agent_name = tool_block["name"]
            tool_use_id = tool_block["toolUseId"]
            task_description = tool_block["input"]["task"]

            # Skip agents already called — prevents Nova Lite loop
            if agent_name in called_agents:
                logger.warning(f"Skipping duplicate call to {agent_name}")
                tool_results.append({"toolResult": {"toolUseId": tool_use_id, "content": [{"text": '{"status": "skipped", "message": "Already completed"}'}]}})
                continue

            called_agents.add(agent_name)
            new_calls = True
            logger.info(f"Orchestrator → [{agent_name.upper()}]: {task_description}")

            agent_key = agent_name.replace("_agent", "")
            agent = agents.get(agent_key)

            if agent:
                result = agent.run(task_description, session_id=session_id)
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

        tool_result_msg = {"role": "user", "content": tool_results}
        messages.append(tool_result_msg)
        save_message(session_id, tool_result_msg)

        # If no new agents were called this iteration, stop
        if not new_calls:
            break

    return all_agent_results


def run_orchestrator_streaming(user_prompt: str, region: str = "us-east-1", session_id: str = "default"):
    """
    Generator version of run_orchestrator for UI streaming.
    Yields events:
      {kind: agent_start,  agent, task}
      {kind: tool_call,    tool, inputs}
      {kind: tool_result,  status, message}
      {kind: agent_done,   agent, status, message}
      {kind: summary,      text}
    """
    ensure_table_exists()
    bedrock = get_bedrock_client()
    agents  = {key: cls(region=region) for key, cls in AGENT_REGISTRY.items()}

    history = load_history(session_id)
    # Keep only last 6 messages to avoid max_tokens on orchestrator
    history = history[-6:] if len(history) > 6 else history
    new_msg = {"role": "user", "content": [{"text": user_prompt}]}
    messages = history + [new_msg]
    save_message(session_id, new_msg)

    called_agents = set()
    all_agent_results = []

    for _ in range(5):
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": ORCHESTRATOR_PROMPT}],
            messages=messages,
            toolConfig={"tools": ORCHESTRATOR_TOOLS},
        )
        output_message = response["output"]["message"]
        messages.append(output_message)
        save_message(session_id, output_message)
        stop_reason = response["stopReason"]

        if stop_reason == "end_turn":
            summary = next(
                (b["text"] for b in output_message.get("content", []) if "text" in b), ""
            )
            import re
            summary = re.sub(r"<thinking>.*?</thinking>", "", summary, flags=re.DOTALL)
            summary = re.sub(r"<response>(.*?)</response>", r"\1", summary, flags=re.DOTALL)
            summary = re.sub(r"<end_turn>", "", summary).strip()

            # If model ended without calling any agent, force-delegate based on keywords
            if not all_agent_results and not called_agents:
                forced = _force_delegate(user_prompt, agents, session_id)
                if forced:
                    all_agent_results.extend(forced)
                    summary = "\n".join(f"{r['agent'].upper()}: {r['result']['message']}" for r in forced)

            if not summary and all_agent_results:
                summary = "\n".join(f"{r['agent'].upper()}: {r['result']['message']}" for r in all_agent_results)
            yield {"kind": "summary", "text": summary or "Done."}
            return

        if stop_reason != "tool_use":
            logger.warning(f"Streaming orchestrator unexpected stop: {stop_reason}")
            if not all_agent_results and not called_agents:
                if stop_reason == "max_tokens":
                    # Context too large — retry with just the user prompt, no history
                    logger.info("max_tokens hit — retrying without history")
                    forced = _force_delegate(user_prompt, agents, session_id=None)
                else:
                    forced = _force_delegate(user_prompt, agents, session_id)
                if forced:
                    all_agent_results.extend(forced)
            if all_agent_results:
                summary = "\n".join(f"{r['agent'].upper()}: {r['result']['message']}" for r in all_agent_results)
                yield {"kind": "summary", "text": summary}
            return

        tool_results = []
        new_calls = False
        for block in output_message.get("content", []):
            if block.get("type") != "toolUse" and "toolUse" not in block:
                continue

            tool_block       = block.get("toolUse") or block
            agent_name       = tool_block["name"]
            tool_use_id      = tool_block["toolUseId"]
            task_description = tool_block["input"]["task"]
            agent_key        = agent_name.replace("_agent", "")
            agent            = agents.get(agent_key)

            # Skip duplicate agent calls
            if agent_name in called_agents:
                logger.warning(f"Skipping duplicate call to {agent_name}")
                tool_results.append({"toolResult": {"toolUseId": tool_use_id, "content": [{"text": '{"status": "skipped", "message": "Already completed"}'}]}})
                continue

            called_agents.add(agent_name)
            new_calls = True

            yield {"kind": "agent_start", "agent": agent_name, "task": task_description}

            if agent:
                final_result = None
                for event in agent.run_streaming(task_description, session_id=session_id):
                    if event["kind"] == "done":
                        final_result = event["result"]
                    else:
                        yield event
                result = final_result or {"status": "error", "message": "No result"}
            else:
                result = {"status": "error", "message": f"Unknown agent: {agent_name}"}

            yield {"kind": "agent_done", "agent": agent_name, "status": result["status"], "message": result["message"]}
            all_agent_results.append({"agent": agent_name, "result": result})

            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps(result, default=str)}],
                }
            })

        tool_result_msg = {"role": "user", "content": tool_results}
        messages.append(tool_result_msg)
        save_message(session_id, tool_result_msg)

        if not new_calls:
            if all_agent_results:
                summary = "\n".join(f"{r['agent'].upper()}: {r['result']['message']}" for r in all_agent_results)
                yield {"kind": "summary", "text": summary}
            return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*", help="Natural language request")
    parser.add_argument("--session", default="default", help="Session ID for memory (default: 'default')")
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else input("Enter your AWS request: ")
    region = os.getenv("AWS_REGION", "ap-south-1")

    results = run_orchestrator(prompt, region, session_id=args.session)

    print("\n===== MULTI-AGENT RESULTS =====")
    for r in results:
        print(f"\n[{r['agent'].upper()}] Task: {r['task']}")
        print(json.dumps(r["result"], indent=2, default=str))
