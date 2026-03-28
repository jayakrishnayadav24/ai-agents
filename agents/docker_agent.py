import subprocess
import json
import os
from agents.base_agent import BaseAgent


class DockerAgent(BaseAgent):
    AGENT_KEY = "docker"
    SYSTEM_PROMPT = """You are a Docker specialist agent running on the local machine.
You manage containers, images, volumes, networks, and docker-compose stacks.

RULES:
- For RUN/CREATE: always pull the image first if not present, then run the container.
- Known required env vars: mysql/mariadb needs MYSQL_ROOT_PASSWORD, postgres needs POSTGRES_PASSWORD.
- If a container already exists with the same name, remove it first then recreate.
- For TROUBLESHOOT: use docker__inspect_container and docker__container_logs to diagnose issues.
- For docker-compose: use docker__compose_up, docker__compose_down, docker__compose_logs.
- Never ask for missing info — use sensible defaults (bridge network, latest tag, etc.).
- If a container name is not given, generate one from the image name.
- Always act autonomously and completely."""

    CAPABILITIES = [
        {
            "name": "list_containers",
            "description": "List all containers (running and stopped)",
            "input_schema": {"type": "object", "properties": {"all": {"type": "boolean", "description": "Include stopped containers"}}},
        },
        {
            "name": "run_container",
            "description": "Pull image if needed and run a new container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "name": {"type": "string"},
                    "ports": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['8080:80', '443:443']"},
                    "volumes": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['/host/path:/container/path']"},
                    "env": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['KEY=VALUE']"},
                    "network": {"type": "string"},
                    "detach": {"type": "boolean"},
                    "command": {"type": "string"},
                },
                "required": ["image"],
            },
        },
        {
            "name": "stop_container",
            "description": "Stop a running container",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "start_container",
            "description": "Start a stopped container",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "remove_container",
            "description": "Remove a container (force stops if running)",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "restart_container",
            "description": "Restart a container",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "container_logs",
            "description": "Get logs from a container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tail": {"type": "integer", "description": "Number of lines from end, default 50"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "inspect_container",
            "description": "Inspect a container — state, ports, mounts, network, env vars",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "exec_container",
            "description": "Execute a command inside a running container",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["name", "command"],
            },
        },
        {
            "name": "list_images",
            "description": "List all local Docker images",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "pull_image",
            "description": "Pull a Docker image from registry",
            "input_schema": {"type": "object", "properties": {"image": {"type": "string"}}, "required": ["image"]},
        },
        {
            "name": "remove_image",
            "description": "Remove a Docker image",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "force": {"type": "boolean"},
                },
                "required": ["image"],
            },
        },
        {
            "name": "list_volumes",
            "description": "List all Docker volumes",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_volume",
            "description": "Create a Docker volume",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "remove_volume",
            "description": "Remove a Docker volume",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "list_networks",
            "description": "List all Docker networks",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_network",
            "description": "Create a Docker network",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "driver": {"type": "string", "description": "bridge, overlay, host, none"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "remove_network",
            "description": "Remove a Docker network",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "compose_up",
            "description": "Run docker-compose up in a directory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to directory containing docker-compose.yml"},
                    "detach": {"type": "boolean"},
                    "build": {"type": "boolean"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "compose_down",
            "description": "Run docker-compose down in a directory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "volumes": {"type": "boolean", "description": "Also remove volumes"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "compose_logs",
            "description": "Get logs from a docker-compose stack",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "tail": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "compose_ps",
            "description": "List containers in a docker-compose stack",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        {
            "name": "system_info",
            "description": "Get Docker system info — version, containers count, images count, disk usage",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "system_prune",
            "description": "Remove all stopped containers, unused networks, dangling images",
            "input_schema": {"type": "object", "properties": {"volumes": {"type": "boolean", "description": "Also prune volumes"}}},
        },
    ]

    def __init__(self, region="us-east-1"):
        super().__init__("DockerAgent", region)

    def _run(self, cmd: list) -> tuple:
        """Run a shell command, return (stdout, stderr, returncode)."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    def execute(self, task: dict) -> dict:
        action = task.get("action")
        if action == "list_containers":    return self.list_containers(task.get("all", True))
        elif action == "run_container":    return self.run_container(task)
        elif action == "stop_container":   return self.stop_container(task["name"])
        elif action == "start_container":  return self.start_container(task["name"])
        elif action == "remove_container": return self.remove_container(task["name"])
        elif action == "restart_container": return self.restart_container(task["name"])
        elif action == "container_logs":   return self.container_logs(task["name"], task.get("tail", 50))
        elif action == "inspect_container": return self.inspect_container(task["name"])
        elif action == "exec_container":   return self.exec_container(task["name"], task["command"])
        elif action == "list_images":      return self.list_images()
        elif action == "pull_image":       return self.pull_image(task["image"])
        elif action == "remove_image":     return self.remove_image(task["image"], task.get("force", False))
        elif action == "list_volumes":     return self.list_volumes()
        elif action == "create_volume":    return self.create_volume(task["name"])
        elif action == "remove_volume":    return self.remove_volume(task["name"])
        elif action == "list_networks":    return self.list_networks()
        elif action == "create_network":   return self.create_network(task["name"], task.get("driver", "bridge"))
        elif action == "remove_network":   return self.remove_network(task["name"])
        elif action == "compose_up":       return self.compose_up(task["path"], task.get("detach", True), task.get("build", False))
        elif action == "compose_down":     return self.compose_down(task["path"], task.get("volumes", False))
        elif action == "compose_logs":     return self.compose_logs(task["path"], task.get("tail", 50))
        elif action == "compose_ps":       return self.compose_ps(task["path"])
        elif action == "system_info":      return self.system_info()
        elif action == "system_prune":     return self.system_prune(task.get("volumes", False))
        return self.report("error", f"Unknown Docker action: {action}")

    def list_containers(self, all_containers: bool = True) -> dict:
        cmd = ["docker", "ps", "--format", "{{json .}}"]
        if all_containers:
            cmd.insert(2, "-a")
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        containers = [json.loads(line) for line in out.splitlines() if line]
        return self.report("success", f"Found {len(containers)} containers", {"containers": containers})

    def run_container(self, task: dict) -> dict:
        image = task["image"]
        # Pull first
        self.pull_image(image)
        cmd = ["docker", "run"]
        if task.get("detach", True):
            cmd.append("-d")
        if task.get("name"):
            cmd += ["--name", task["name"]]
        for p in task.get("ports", []):
            cmd += ["-p", p]
        for v in task.get("volumes", []):
            cmd += ["-v", v]
        for e in task.get("env", []):
            cmd += ["-e", e]
        if task.get("network"):
            cmd += ["--network", task["network"]]
        cmd.append(image)
        if task.get("command"):
            cmd += task["command"].split()
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        return self.report("created", f"Container started: {out[:12]}", {"container_id": out})

    def stop_container(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "stop", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Container '{name}' stopped")

    def start_container(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "start", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Container '{name}' started")

    def remove_container(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "rm", "-f", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("deleted", f"Container '{name}' removed")

    def restart_container(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "restart", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Container '{name}' restarted")

    def container_logs(self, name: str, tail: int = 50) -> dict:
        out, err, rc = self._run(["docker", "logs", "--tail", str(tail), name])
        logs = out or err
        return self.report("success", f"Logs for '{name}'", {"logs": logs})

    def inspect_container(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "inspect", name])
        if rc != 0:
            return self.report("error", err)
        data = json.loads(out)[0]
        info = {
            "name": data["Name"].lstrip("/"),
            "status": data["State"]["Status"],
            "running": data["State"]["Running"],
            "image": data["Config"]["Image"],
            "ports": data["NetworkSettings"]["Ports"],
            "mounts": [{"source": m["Source"], "destination": m["Destination"]} for m in data.get("Mounts", [])],
            "networks": list(data["NetworkSettings"]["Networks"].keys()),
            "env": data["Config"].get("Env", []),
            "restart_count": data["RestartCount"],
        }
        return self.report("success", f"Container '{name}' is {info['status']}", info)

    def exec_container(self, name: str, command: str) -> dict:
        cmd = ["docker", "exec", name] + command.split()
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Command executed in '{name}'", {"output": out})

    def list_images(self) -> dict:
        out, err, rc = self._run(["docker", "images", "--format", "{{json .}}"])
        if rc != 0:
            return self.report("error", err)
        images = [json.loads(line) for line in out.splitlines() if line]
        return self.report("success", f"Found {len(images)} images", {"images": images})

    def pull_image(self, image: str) -> dict:
        self.logger.info(f"Pulling image: {image}")
        out, err, rc = self._run(["docker", "pull", image])
        if rc != 0:
            return self.report("error", f"Failed to pull {image}: {err}")
        return self.report("success", f"Image '{image}' pulled")

    def remove_image(self, image: str, force: bool = False) -> dict:
        cmd = ["docker", "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(image)
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        return self.report("deleted", f"Image '{image}' removed")

    def list_volumes(self) -> dict:
        out, err, rc = self._run(["docker", "volume", "ls", "--format", "{{json .}}"])
        if rc != 0:
            return self.report("error", err)
        volumes = [json.loads(line) for line in out.splitlines() if line]
        return self.report("success", f"Found {len(volumes)} volumes", {"volumes": volumes})

    def create_volume(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "volume", "create", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("created", f"Volume '{name}' created")

    def remove_volume(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "volume", "rm", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("deleted", f"Volume '{name}' removed")

    def list_networks(self) -> dict:
        out, err, rc = self._run(["docker", "network", "ls", "--format", "{{json .}}"])
        if rc != 0:
            return self.report("error", err)
        networks = [json.loads(line) for line in out.splitlines() if line]
        return self.report("success", f"Found {len(networks)} networks", {"networks": networks})

    def create_network(self, name: str, driver: str = "bridge") -> dict:
        out, err, rc = self._run(["docker", "network", "create", "--driver", driver, name])
        if rc != 0:
            return self.report("error", err)
        return self.report("created", f"Network '{name}' created with driver '{driver}'")

    def remove_network(self, name: str) -> dict:
        out, err, rc = self._run(["docker", "network", "rm", name])
        if rc != 0:
            return self.report("error", err)
        return self.report("deleted", f"Network '{name}' removed")

    def compose_up(self, path: str, detach: bool = True, build: bool = False) -> dict:
        cmd = ["docker", "compose", "-f", os.path.join(path, "docker-compose.yml"), "up"]
        if detach:
            cmd.append("-d")
        if build:
            cmd.append("--build")
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Compose stack at '{path}' started", {"output": out or err})

    def compose_down(self, path: str, volumes: bool = False) -> dict:
        cmd = ["docker", "compose", "-f", os.path.join(path, "docker-compose.yml"), "down"]
        if volumes:
            cmd.append("-v")
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Compose stack at '{path}' stopped")

    def compose_logs(self, path: str, tail: int = 50) -> dict:
        out, err, rc = self._run(["docker", "compose", "-f", os.path.join(path, "docker-compose.yml"),
                                   "logs", "--tail", str(tail)])
        return self.report("success", f"Compose logs from '{path}'", {"logs": out or err})

    def compose_ps(self, path: str) -> dict:
        out, err, rc = self._run(["docker", "compose", "-f", os.path.join(path, "docker-compose.yml"), "ps"])
        if rc != 0:
            return self.report("error", err)
        return self.report("success", f"Compose services at '{path}'", {"output": out})

    def system_info(self) -> dict:
        out, err, rc = self._run(["docker", "system", "info", "--format", "{{json .}}"])
        if rc != 0:
            return self.report("error", err)
        info = json.loads(out)
        return self.report("success", "Docker system info", {
            "containers": info.get("Containers"),
            "running": info.get("ContainersRunning"),
            "stopped": info.get("ContainersStopped"),
            "images": info.get("Images"),
            "server_version": info.get("ServerVersion"),
            "os": info.get("OperatingSystem"),
        })

    def system_prune(self, volumes: bool = False) -> dict:
        cmd = ["docker", "system", "prune", "-f"]
        if volumes:
            cmd.append("--volumes")
        out, err, rc = self._run(cmd)
        if rc != 0:
            return self.report("error", err)
        return self.report("success", "Docker system pruned", {"output": out})
