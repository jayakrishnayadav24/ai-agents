# AWS Multi-Agent System

A true agentic AI system that takes natural language prompts and autonomously manages AWS infrastructure and local Docker using multiple specialist AI agents powered by Amazon Bedrock.

---

## Why This Was Built

The idea behind this project came from a real frustration — managing AWS infrastructure requires jumping between EC2, ECS, EKS, and IAM consoles, running multiple CLI commands, and knowing the exact sequence of steps. For example, to create an ECS Fargate service you need to:

1. Check if the IAM execution role exists
2. Create it if missing
3. Create the ECS cluster
4. Register a task definition
5. Find real subnet IDs from your VPC
6. Create the service with the right network config

That's 6 steps, and if any one fails you have to debug and retry manually.

The same problem exists with Docker — spinning up multiple containers with volumes, networks, and the right env vars requires remembering exact commands and the right order.

The goal was simple — **just say what you want in plain English and let AI agents figure out the steps, fix errors, and get it done.**

```bash
python3 main.py "Create an ECS Fargate cluster named my-app with nginx on port 80"
# → Agents automatically handle all 6 steps above

python3 main.py "Run mongo 4, redis 4, and mysql 8.4 with volumes on the same network"
# → Docker agent creates network, pulls images, runs all 3 containers
```

---

## What Makes This "Agentic AI"

This is not just a chatbot or a script. It is a **multi-agent system** where each agent has its own AI brain, its own tools, and its own reasoning loop.

### Evolution During Development

**Version 1 — Single LLM with tools (tool-use AI)**
```
User → One Nova LLM → picks tools → EC2/ECS/EKS/IAM functions
```
One brain, multiple tools. The LLM decided the sequence but all agents were just Python functions with no intelligence of their own.

**Version 2 — True Multi-Agent (current)**
```
User → Orchestrator LLM → delegates → EC2 Agent LLM    (own brain)
                        → delegates → ECS Agent LLM    (own brain)
                        → delegates → EKS Agent LLM    (own brain)
                        → delegates → IAM Agent LLM    (own brain)
                        → delegates → Docker Agent LLM (own brain)
```
Each agent has its own Amazon Bedrock call, its own system prompt, its own tools, and its own reasoning loop. The orchestrator just delegates — it does not micromanage.

---

## Architecture

```
User Prompt (natural language)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│              Orchestrator LLM (main.py)               │
│                                                       │
│  Has 5 tools: ec2_agent, ecs_agent, eks_agent,        │
│               iam_agent, docker_agent                 │
│                                                       │
│  Decides WHICH agents to call and WHAT task           │
│  to give each one. Does not decide HOW.               │
└──────────────────────────────────────────────────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│  EC2   │ │  ECS   │ │  EKS   │ │  IAM   │ │Docker  │
│ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │
│own LLM │ │own LLM │ │own LLM │ │own LLM │ │own LLM │
│own tools│ │own tools│ │own tools│ │own tools│ │own tools│
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
     │                                            │
     ▼                                            ▼
AWS APIs                                   Local Docker CLI
(EC2/ECS/EKS/IAM)                         (containers/images/
                                           volumes/networks)
```

### How a Request Flows

```
"Run mongo 4, redis 4, mysql 8.4 with volumes on same network"
        │
        ▼
Orchestrator LLM thinks:
  "This is a Docker task, delegate to docker_agent"
        │
        └──► Docker Agent LLM:
               "Create network → pull images → run containers"
               → calls docker__create_network(app-network)
               → calls docker__pull_image(mongo:4)
               → calls docker__pull_image(redis:4)
               → calls docker__pull_image(mysql:8.4)
               → calls docker__run_container(mongo:4, volume, network)
               → calls docker__run_container(redis:4, volume, network)
               → calls docker__run_container(mysql:8.4, MYSQL_ROOT_PASSWORD, volume, network)
               → reports: "All 3 containers running on app-network"
        │
        ▼
Orchestrator summarizes final result to user
```

No hardcoded steps. Each agent reasons independently.

---

## Project Structure

```
aws-multi-agent/
├── main.py                  # Orchestrator — delegates to specialist agents
├── requirements.txt         # boto3, botocore
└── agents/
    ├── base_agent.py        # BaseAgent — shared LLM loop, tool dispatch
    ├── iam_agent.py         # IAM specialist — users, groups, roles, policies
    ├── ec2_agent.py         # EC2 specialist — instances, VPC, SGs
    ├── ecs_agent.py         # ECS specialist — clusters, services, tasks
    ├── eks_agent.py         # EKS specialist — clusters, nodegroups
    └── docker_agent.py      # Docker specialist — containers, images, volumes, networks, compose
```

---

## Code Explanation

### `base_agent.py` — The Brain of Every Agent

Every agent extends `BaseAgent`. This is where the agentic loop lives.

```python
class BaseAgent:
    AGENT_KEY: str        # e.g. "ecs"
    CAPABILITIES: list    # list of tools this agent has
    SYSTEM_PROMPT: str    # this agent's personality and rules
```

Key methods:
- `run(task_description)` — the **agentic loop**: sends task to Bedrock, gets tool calls back, executes them, feeds results back, repeats until done
- `execute(task)` — direct API/CLI call, no LLM involved
- `get_tools()` — converts CAPABILITIES into Bedrock tool specs
- `_get_dispatcher()` — maps tool names to callable functions
- `bedrock` property — lazy-loads a working Bedrock client, auto-tries multiple profiles

### `main.py` — The Orchestrator

The orchestrator is itself an LLM agent. It has 5 tools — one per specialist agent. When it calls `docker_agent`, it's not calling a function — it's delegating to another LLM that will reason and act independently.

```python
AGENT_REGISTRY = {
    "ec2": EC2Agent,
    "ecs": ECSAgent,
    "eks": EKSAgent,
    "iam": IAMAgent,
    "docker": DockerAgent,
}
```

Adding a new agent = create the file + add one line here.

---

## Available Tools (72 total across 6 agents)

### IAMAgent (15 tools)
| Tool | Description |
|------|-------------|
| `iam__create_user` | Create an IAM user |
| `iam__list_users` | List all IAM users |
| `iam__delete_user` | Delete an IAM user (detaches policies, removes from groups) |
| `iam__attach_user_policy` | Attach a managed policy directly to a user |
| `iam__detach_user_policy` | Detach a managed policy from a user |
| `iam__list_user_policies` | List all policies attached to a user |
| `iam__create_access_key` | Create access key for a user |
| `iam__create_group` | Create an IAM group |
| `iam__list_groups` | List all IAM groups |
| `iam__add_user_to_group` | Add a user to a group |
| `iam__attach_group_policy` | Attach a managed policy to a group |
| `iam__check_role` | Check if an IAM role exists |
| `iam__create_role` | Create an IAM role with trust policy and managed policies |
| `iam__list_roles` | List all IAM roles |
| `iam__list_policies` | List AWS managed policies filtered by keyword |

### EC2Agent (11 tools)
| Tool | Description |
|------|-------------|
| `ec2__create_instance` | Launch a new EC2 instance |
| `ec2__start_instance` | Start a stopped instance |
| `ec2__stop_instance` | Stop a running instance |
| `ec2__describe_instances` | Describe instances with filters |
| `ec2__diagnose_instance` | Full diagnosis — state, status checks, SG, IP, subnet |
| `ec2__list_vpcs` | List all VPCs with CIDR and default flag |
| `ec2__diagnose_vpc` | Check IGW, NAT, route tables, public/private subnets |
| `ec2__list_subnets` | List all subnets with AZ and available IPs |
| `ec2__list_security_groups` | List all security groups |
| `ec2__check_security_group_rules` | Show all inbound/outbound rules of a SG |
| `ec2__fix_security_group` | Add ingress rules to a security group |

### ECSAgent (12 tools)
| Tool | Description |
|------|-------------|
| `ecs__create_cluster` | Create an ECS cluster with Container Insights |
| `ecs__list_clusters` | List all ECS clusters |
| `ecs__describe_cluster` | Show cluster status, running/pending tasks |
| `ecs__register_task` | Register a task definition (auto-creates execution role) |
| `ecs__create_service` | Create a Fargate service (auto-creates service linked role) |
| `ecs__list_services` | List all services in a cluster |
| `ecs__describe_services` | Describe specific services |
| `ecs__diagnose_service` | Full diagnosis — counts, events, deployments |
| `ecs__fix_stopped_tasks` | Get stop reasons for all stopped tasks |
| `ecs__list_running_tasks` | List running tasks with container status |
| `ecs__delete_service` | Scale to 0 and force delete a service |
| `ecs__delete_cluster` | Delete an ECS cluster |

### EKSAgent (10 tools)
| Tool | Description |
|------|-------------|
| `eks__list_clusters` | List all EKS clusters |
| `eks__create_cluster` | Create cluster (auto-creates IAM role, handles already-exists) |
| `eks__describe_cluster` | Show status, version, endpoint, logging |
| `eks__wait_for_cluster` | Poll until cluster is ACTIVE (max 20 min) |
| `eks__diagnose_cluster` | Full diagnosis — status, nodegroups, health issues |
| `eks__create_nodegroup` | Create managed nodegroup (auto-creates node IAM role) |
| `eks__list_nodegroups` | List all nodegroups in a cluster |
| `eks__describe_nodegroup` | Show status, scaling config, health issues |
| `eks__fix_nodegroup` | Update scaling config of a nodegroup |
| `eks__list_supported_versions` | Get supported k8s versions in this region |

### DockerAgent (24 tools)
| Tool | Description |
|------|-------------|
| `docker__list_containers` | List all containers (running and stopped) |
| `docker__run_container` | Pull image if needed and run a new container |
| `docker__stop_container` | Stop a running container |
| `docker__start_container` | Start a stopped container |
| `docker__remove_container` | Remove a container (force stops if running) |
| `docker__restart_container` | Restart a container |
| `docker__container_logs` | Get logs from a container |
| `docker__inspect_container` | Inspect state, ports, mounts, network, env vars |
| `docker__exec_container` | Execute a command inside a running container |
| `docker__list_images` | List all local Docker images |
| `docker__pull_image` | Pull a Docker image from registry |
| `docker__remove_image` | Remove a Docker image |
| `docker__list_volumes` | List all Docker volumes |
| `docker__create_volume` | Create a Docker volume |
| `docker__remove_volume` | Remove a Docker volume |
| `docker__list_networks` | List all Docker networks |
| `docker__create_network` | Create a Docker network |
| `docker__remove_network` | Remove a Docker network |
| `docker__compose_up` | Run docker-compose up in a directory |
| `docker__compose_down` | Run docker-compose down in a directory |
| `docker__compose_logs` | Get logs from a docker-compose stack |
| `docker__compose_ps` | List containers in a docker-compose stack |
| `docker__system_info` | Get Docker system info — version, counts, disk usage |
| `docker__system_prune` | Remove stopped containers, unused networks, dangling images |

---

## Setup

### 1. Clone and create virtual environment
```bash
cd aws-multi-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure AWS credentials
Edit `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = <your-key>
aws_secret_access_key = <your-secret>

[own]
aws_access_key_id = <bedrock-account-key>
aws_secret_access_key = <bedrock-account-secret>
```

### 3. Set environment variables
```bash
export AWS_PROFILE=test    # profile for creating AWS resources
export AWS_REGION=ap-south-1      # region for AWS resources
export BEDROCK_PROFILE=own        # profile that has Bedrock access
export BEDROCK_REGION=ap-south-1  # region where Bedrock is available
```

### 4. Required IAM permissions
```
ec2:*, ecs:*, eks:*, iam:*, bedrock:InvokeModel
```

### 5. Docker requirement
Docker must be installed and running on your local machine for the Docker agent:
```bash
docker --version   # verify Docker is installed
docker ps          # verify Docker daemon is running
```

---

## Running

```bash
python3 main.py "your natural language request here"
```

Interactive mode:
```bash
python3 main.py
# Enter your AWS request: <type here>
```

---

## Example Queries

### IAM — Users
```bash
# Create user with specific access
python3 main.py "create a iam user with read only access to only ec2"
python3 main.py "create iam user named devops-admin with full admin access"
python3 main.py "create iam user named s3reader with read only access to S3"

# List and manage
python3 main.py "list all iam users"
python3 main.py "delete iam user named ec2-readonly"
python3 main.py "list all EC2 related AWS managed policies"
```

### IAM — Roles
```bash
python3 main.py "Check if ecsTaskExecutionRole exists"
python3 main.py "Check if ecsTaskExecutionRole exists, create it if not"
python3 main.py "Check if ecsTaskExecutionRole and eksClusterRole exist, create if missing"
```

### EC2 — Create & Manage
```bash
python3 main.py "Create a ec2 for me with default config"
python3 main.py "Create a t3.medium EC2 instance named web-server"
python3 main.py "Start EC2 instance i-0abc123def456"
python3 main.py "Stop EC2 instance i-0abc123def456"
```

<img width="1918" height="940" alt="image" src="https://github.com/user-attachments/assets/c46994a1-6c96-4663-aa1a-08378eb96d4a" />


### EC2 — Troubleshoot
```bash
python3 main.py "Diagnose EC2 instance i-0abc123def456"
python3 main.py "My EC2 instance i-0abc123 is unreachable, diagnose it"
python3 main.py "Show me all rules for security group sg-0abc123"
python3 main.py "Open port 443 on security group sg-0abc123"
python3 main.py "List all my running EC2 instances"
```

### VPC — Troubleshoot
```bash
python3 main.py "List all my VPCs"
python3 main.py "Diagnose VPC vpc-05bc5283de6e92427 for connectivity issues"
python3 main.py "Does my VPC have an internet gateway?"
python3 main.py "Which subnets are public and which are private?"
python3 main.py "Check my default VPC for any connectivity or routing issues"
```

### ECS — Create
```bash
python3 main.py "Create an ECS Fargate cluster named my-app with nginx on port 80"
python3 main.py "Create ECS cluster prod with httpd:latest on port 8080"
python3 main.py "Create ECS cluster staging with 3 nginx replicas on port 80"
```

### ECS — Troubleshoot
```bash
python3 main.py "Diagnose ECS service nginx-service in cluster my-app"
python3 main.py "My ECS tasks in cluster prod keep stopping, diagnose and fix"
python3 main.py "List all running tasks in my prod ECS cluster"
python3 main.py "Describe my ECS cluster named my-app"
python3 main.py "List all my ECS clusters"
```

### ECS — Delete
```bash
python3 main.py "Delete ECS service nginx-service from cluster my-app"
python3 main.py "Delete ECS cluster my-app and everything in it"
```

### EKS — Create
```bash
python3 main.py "Create an EKS cluster named dev-cluster with 2 t3.medium nodes"
python3 main.py "Create EKS cluster prod-cluster with 3 t3.large nodes"
python3 main.py "Add a nodegroup named gpu-nodes in EKS cluster dev-cluster with 2 g4dn.xlarge instances"
```

### EKS — Troubleshoot
```bash
python3 main.py "Diagnose my EKS cluster dev-cluster"
python3 main.py "Check the health of nodegroup default-ng in EKS cluster dev-cluster"
python3 main.py "List all my EKS clusters"
python3 main.py "Scale up nodegroup default-ng in dev-cluster to 5 nodes"
```

### EKS — Delete
```bash
python3 main.py "Delete EKS cluster dev-cluster"
python3 main.py "Delete all EKS clusters"
```

### Docker — Containers
```bash
# Run containers
python3 main.py "Run nginx on port 8080"
python3 main.py "Run postgres with a volume named pgdata on port 5432 with password mypassword"
python3 main.py "Run mongo version 4 with a volume attached"

# Multi-container setup on same network
python3 main.py "Run mongo 4, redis 4, and mysql 8.4 with volumes all on the same network named app-network"

# Manage
python3 main.py "Stop container named nginx"
python3 main.py "Restart my mongo container"
python3 main.py "Remove container named old-app"
python3 main.py "List all running containers"
python3 main.py "List all containers including stopped ones"
```

### Docker — Troubleshoot
```bash
python3 main.py "My nginx container keeps restarting, diagnose it"
python3 main.py "Show me logs of my mongo container"
python3 main.py "Show last 100 lines of logs from mysql container"
python3 main.py "Inspect my redis container — show ports, volumes, network"
python3 main.py "Run a command inside my nginx container: nginx -t"
```

### Docker — Images
```bash
python3 main.py "Pull the latest postgres image"
python3 main.py "List all my local docker images"
python3 main.py "Remove the old nginx image"
```

### Docker — Volumes & Networks
```bash
python3 main.py "Create a docker volume named mydata"
python3 main.py "List all docker volumes"
python3 main.py "Create a bridge network named backend"
python3 main.py "List all docker networks"
python3 main.py "Remove network named old-network"
```

### Docker — Compose
```bash
python3 main.py "Start the docker-compose stack at /home/jaya/myapp"
python3 main.py "Stop the docker-compose stack at /home/jaya/myapp"
python3 main.py "Show logs from docker-compose stack at /home/jaya/myapp"
python3 main.py "List services in docker-compose stack at /home/jaya/myapp"
python3 main.py "Start docker-compose at /home/jaya/myapp with build"
```

### Docker — Cleanup
```bash
python3 main.py "Show docker system info"
python3 main.py "Clean up all stopped containers and unused images"
python3 main.py "Prune docker system including volumes"
```

### Complex Multi-Agent Queries
```bash
# ECS + IAM together
python3 main.py "Create ECS cluster prod with nginx on port 80 and check all required IAM roles"

# EKS full setup
python3 main.py "Create EKS cluster dev-cluster with 2 t3.medium nodes and verify everything is healthy"

# Cross-service diagnosis
python3 main.py "My ECS service in prod cluster is failing, check the service, stopped tasks, and security groups"

# IAM + Docker together
python3 main.py "Create an IAM user with EC2 read-only access and also run nginx locally on port 80"

# Full environment setup
python3 main.py "Set up a complete ECS Fargate environment with cluster named api, nginx on port 80, ensure all IAM roles exist"

# Cleanup everything
python3 main.py "Delete ECS cluster prod and all its services"
python3 main.py "Delete all EKS clusters"
python3 main.py "Stop and remove all docker containers and prune images"
```

---

## Pros of This System

### 1. True Natural Language Interface
You don't need to know AWS CLI commands, Docker commands, API parameters, or the exact sequence of steps. Just describe what you want.
```bash
# Instead of 6 CLI commands:
python3 main.py "Create ECS cluster prod with nginx on port 80"

# Instead of docker network create + docker pull x3 + docker run x3:
python3 main.py "Run mongo 4, redis 4, mysql 8.4 with volumes on same network"
```

### 2. Each Agent is a True Specialist
Every agent has its own LLM call, its own system prompt, its own tools, and its own reasoning loop. The Docker agent knows Docker deeply. The EKS agent knows EKS deeply. They don't interfere with each other.

### 3. Autonomous Error Recovery
When something fails, agents don't crash — they reason about the error and try a different approach:
- Subnet not found → agent calls `ec2__list_subnets` to find real ones
- IAM role missing → agent creates it automatically before proceeding
- EKS cluster already exists → agent treats it as success and continues
- Docker container name conflict → agent removes old container and recreates
- MySQL missing password → agent adds `MYSQL_ROOT_PASSWORD` automatically

### 4. Zero Hardcoded Flows
There is no `if action == "create_ecs" then do step1, step2, step3`. The AI decides the sequence based on results.

### 5. Covers Both Cloud and Local
Unlike other tools that only manage cloud resources, this system also manages your local Docker environment — containers, images, volumes, networks, and compose stacks.

### 6. Multi-Account Support
Bedrock (AI inference) runs on one AWS account while resource creation happens on another. The system auto-detects which profile works and falls back gracefully.

### 7. Easily Extensible
Adding a new AWS service or local tool takes 3 steps:
1. Create `agents/your_agent.py` with `CAPABILITIES` list
2. Add one line to `AGENT_REGISTRY` in `main.py`
3. Done — the tool is automatically exposed to the orchestrator

### 8. Real Actions
This is not a simulation. Every action creates/modifies/deletes real AWS resources and real Docker containers on your machine.

---

## Cons and Limitations

### 1. Model Quality Bottleneck
The system currently uses `apac.amazon.nova-lite-v1:0` which is a small, free model. It sometimes:
- Gets stuck in loops calling the same tool repeatedly
- Misunderstands "delete" as "create"
- Stops mid-task and asks for clarification instead of acting autonomously

**Fix:** Use Claude 3.5 Sonnet when Anthropic billing is resolved. Smarter model = fewer loops, better reasoning.

### 2. No Persistent Memory
Each run starts fresh. The agents don't remember previous conversations.

**Fix:** Add DynamoDB or a local JSON file to store session history.

### 3. Sequential Agent Execution
Agents run one at a time. If the orchestrator needs both EC2 and ECS agents, it calls them sequentially.

**Fix:** Use Python `threading` or `asyncio` to run multiple agents simultaneously.

### 4. No Confirmation Before Destructive Actions
If you say "delete all EKS clusters" or "remove all containers", it will do it without asking "are you sure?".

**Fix:** Add a `--dry-run` flag or a confirmation prompt for delete operations.

### 5. Bedrock Payment Dependency
The system depends on AWS Bedrock which requires a valid payment instrument. Free-tier Anthropic models hit payment walls intermittently.

**Fix:** Ensure billing is set up on the Bedrock account, or use a local model via Ollama as fallback.

### 6. No Real-Time Streaming
Results are shown only after the entire agent loop completes. For long operations like EKS cluster creation (10-15 min), you see nothing until it's done.

**Fix:** Add streaming output or periodic status updates.

### 7. Docker Agent Requires Local Docker
The Docker agent runs `docker` CLI commands on your local machine. It won't work if Docker is not installed or the daemon is not running.

### 8. Context Window Limits
For large accounts with hundreds of security groups or subnets, the tool results can exceed the model's context window.

**Fix:** Truncate large list results before feeding back to the model.

---

## Adding a New Agent

1. Create `agents/s3_agent.py`:
```python
from agents.base_agent import BaseAgent

class S3Agent(BaseAgent):
    AGENT_KEY = "s3"
    SYSTEM_PROMPT = """You are an S3 specialist agent.
    - Always use real bucket names.
    - Never delete buckets without listing contents first."""

    CAPABILITIES = [
        {
            "name": "list_buckets",
            "description": "List all S3 buckets",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_bucket",
            "description": "Create an S3 bucket",
            "input_schema": {
                "type": "object",
                "properties": {"bucket_name": {"type": "string"}},
                "required": ["bucket_name"],
            },
        },
    ]

    def __init__(self, region="us-east-1"):
        super().__init__("S3Agent", region)
        self.s3 = self.session.client("s3")

    def execute(self, task: dict) -> dict:
        action = task.get("action")
        if action == "list_buckets":    return self.list_buckets()
        elif action == "create_bucket": return self.create_bucket(task["bucket_name"])
        return self.report("error", f"Unknown S3 action: {action}")

    def list_buckets(self) -> dict:
        try:
            buckets = [b["Name"] for b in self.s3.list_buckets()["Buckets"]]
            return self.report("success", f"Found {len(buckets)} buckets", {"buckets": buckets})
        except Exception as e:
            return self.report("error", str(e))

    def create_bucket(self, bucket_name: str) -> dict:
        try:
            self.s3.create_bucket(Bucket=bucket_name)
            return self.report("created", f"Bucket '{bucket_name}' created")
        except Exception as e:
            return self.report("error", str(e))
```

2. Register in `main.py`:
```python
from agents.s3_agent import S3Agent

AGENT_REGISTRY = {
    "ec2": EC2Agent,
    "ecs": ECSAgent,
    "eks": EKSAgent,
    "iam": IAMAgent,
    "docker": DockerAgent,
    "s3": S3Agent,   # add this
}

# Also add to ORCHESTRATOR_TOOLS:
{
    "toolSpec": {
        "name": "s3_agent",
        "description": "Delegate a task to the S3 specialist agent.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            }
        },
    }
},
```

That's it. The S3 agent is now a full specialist with its own LLM reasoning.

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_PROFILE` | `default` | AWS profile for creating resources |
| `AWS_REGION` | `ap-south-1` | Region where resources are created |
| `BEDROCK_PROFILE` | `own` | AWS profile that has Bedrock access |
| `BEDROCK_REGION` | `ap-south-1` | Region where Bedrock is available |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AccessDeniedException: INVALID_PAYMENT_INSTRUMENT` | Add valid payment method to the Bedrock account in AWS Billing console |
| `All Bedrock profiles failed` | Check `~/.aws/credentials` has valid keys for `own`/`default` profiles |
| `ClusterNotFoundException` | Cluster doesn't exist — check `AWS_PROFILE` is pointing to the right account |
| `InvalidSubnetID.NotFound` | Subnet doesn't exist — agents auto-call `ec2__list_subnets` to find real ones |
| `ResourceInUseException: Cluster already exists` | Agents handle this automatically — treated as success, proceeds to next step |
| Agent loops on same tool | Model quality issue — switch to Claude 3.5 Sonnet when billing is resolved |
| `max_tokens` stop reason | Too many results in context — reduce list sizes or upgrade to larger context model |
| `docker: command not found` | Install Docker on your machine and ensure the daemon is running |
| MySQL container exits immediately | Agent now auto-adds `MYSQL_ROOT_PASSWORD` — if it still fails, check Docker logs |
| `Cannot connect to Docker daemon` | Run `sudo systemctl start docker` or start Docker Desktop |
