from agents.base_agent import BaseAgent


class EC2Agent(BaseAgent):
    AGENT_KEY = "ec2"
    SYSTEM_PROMPT = """You are an EC2 specialist agent. You manage EC2 instances, VPCs, subnets, and security groups.
- Always use real IDs — call ec2__list_subnets or ec2__list_security_groups to discover them if not provided.
- For CREATE: never ask the user for details. Act autonomously with these defaults if not specified:
  * ami_id: ami-0f58b397bc5c1f2e8 (Amazon Linux 2, ap-south-1)
  * instance_type: t2.micro
  * name: auto-created
  * subnet_id: pick the first subnet from ec2__list_subnets
  * security_group_ids: pick the group named 'default' from ec2__list_security_groups
  * key_name: omit entirely if not provided
- Just do it — never ask for confirmation or missing details. Proceed autonomously.
- For DIAGNOSE: use ec2__diagnose_instance and ec2__diagnose_vpc.
- For DELETE: only use stop/terminate tools, never create tools."""

    CAPABILITIES = [
        {
            "name": "create_instance",
            "description": "Launch a new EC2 instance",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ami_id": {"type": "string"}, "instance_type": {"type": "string"},
                    "name": {"type": "string"}, "subnet_id": {"type": "string"},
                    "key_name": {"type": "string"},
                    "security_group_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["ami_id", "instance_type", "name"],
            },
        },
        {
            "name": "start_instance",
            "description": "Start a stopped EC2 instance",
            "input_schema": {"type": "object", "properties": {"instance_id": {"type": "string"}}, "required": ["instance_id"]},
        },
        {
            "name": "stop_instance",
            "description": "Stop a running EC2 instance",
            "input_schema": {"type": "object", "properties": {"instance_id": {"type": "string"}}, "required": ["instance_id"]},
        },
        {
            "name": "describe_instances",
            "description": "Describe EC2 instances with optional filters",
            "input_schema": {"type": "object", "properties": {"filters": {"type": "array", "items": {"type": "object"}}}},
        },
        {
            "name": "diagnose_instance",
            "description": "Diagnose an EC2 instance — state, status checks, security groups, IP, subnet, VPC",
            "input_schema": {"type": "object", "properties": {"instance_id": {"type": "string"}}, "required": ["instance_id"]},
        },
        {
            "name": "list_vpcs",
            "description": "List all VPCs with CIDR, state, and default flag",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "diagnose_vpc",
            "description": "Diagnose a VPC — IGW, NAT, route tables, public/private subnets",
            "input_schema": {"type": "object", "properties": {"vpc_id": {"type": "string"}}, "required": ["vpc_id"]},
        },
        {
            "name": "list_subnets",
            "description": "List all subnets with IDs, VPC, AZ, and available IPs",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_security_groups",
            "description": "List all security groups in the account",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "check_security_group_rules",
            "description": "Show all inbound and outbound rules of a security group",
            "input_schema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]},
        },
        {
            "name": "fix_security_group",
            "description": "Add ingress rules to an EC2 security group",
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "ip_permissions": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["group_id", "ip_permissions"],
            },
        },
    ]

    def __init__(self, region="us-east-1"):
        super().__init__("EC2Agent", region)
        self.ec2 = self.session.client("ec2")

    def execute(self, task: dict) -> dict:
        action = task.get("action")
        if action == "create_instance":          return self.create_instance(task)
        elif action == "stop_instance":          return self.stop_instance(task["instance_id"])
        elif action == "start_instance":         return self.start_instance(task["instance_id"])
        elif action == "describe_instances":     return self.describe_instances(task.get("filters", []))
        elif action == "diagnose_instance":      return self.diagnose_instance(task["instance_id"])
        elif action == "list_vpcs":              return self.list_vpcs()
        elif action == "diagnose_vpc":           return self.diagnose_vpc(task["vpc_id"])
        elif action == "list_subnets":           return self.list_subnets()
        elif action == "list_security_groups":   return self.list_security_groups()
        elif action == "check_security_group_rules": return self.check_security_group_rules(task["group_id"])
        elif action == "fix_security_group":     return self.fix_security_group(task)
        return self.report("error", f"Unknown EC2 action: {action}")

    def create_instance(self, task: dict) -> dict:
        try:
            kwargs = {
                "ImageId": task["ami_id"],
                "InstanceType": task.get("instance_type", "t2.micro"),
                "MinCount": 1, "MaxCount": 1,
                "SecurityGroupIds": task.get("security_group_ids", []),
                "SubnetId": task.get("subnet_id"),
                "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": task.get("name", "auto-created")}]}],
            }
            if task.get("key_name"):  # only add KeyName if explicitly provided and non-empty
                kwargs["KeyName"] = task["key_name"]
            resp = self.ec2.run_instances(**kwargs)
            iid = resp["Instances"][0]["InstanceId"]
            return self.report("created", f"EC2 instance created: {iid}", {"instance_id": iid})
        except Exception as e:
            return self.report("error", str(e))

    def stop_instance(self, instance_id: str) -> dict:
        try:
            self.ec2.stop_instances(InstanceIds=[instance_id])
            return self.report("success", f"Instance {instance_id} stopped")
        except Exception as e:
            return self.report("error", str(e))

    def start_instance(self, instance_id: str) -> dict:
        try:
            self.ec2.start_instances(InstanceIds=[instance_id])
            return self.report("success", f"Instance {instance_id} started")
        except Exception as e:
            return self.report("error", str(e))

    def describe_instances(self, filters: list) -> dict:
        try:
            resp = self.ec2.describe_instances(Filters=filters)
            instances = [i for r in resp["Reservations"] for i in r["Instances"]]
            return self.report("success", f"Found {len(instances)} instances", {"instances": instances})
        except Exception as e:
            return self.report("error", str(e))

    def diagnose_instance(self, instance_id: str) -> dict:
        try:
            inst = self.ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
            info = {
                "state": inst["State"]["Name"], "instance_type": inst["InstanceType"],
                "public_ip": inst.get("PublicIpAddress"), "private_ip": inst.get("PrivateIpAddress"),
                "subnet_id": inst.get("SubnetId"), "vpc_id": inst.get("VpcId"),
                "security_groups": [sg["GroupId"] for sg in inst.get("SecurityGroups", [])],
                "launch_time": str(inst.get("LaunchTime", "")),
            }
            status = self.ec2.describe_instance_status(InstanceIds=[instance_id], IncludeAllInstances=True)
            if status["InstanceStatuses"]:
                s = status["InstanceStatuses"][0]
                info["system_status"] = s["SystemStatus"]["Status"]
                info["instance_status"] = s["InstanceStatus"]["Status"]
            return self.report("success", f"Instance {instance_id} is {info['state']}", info)
        except Exception as e:
            return self.report("error", str(e))

    def list_vpcs(self) -> dict:
        try:
            vpcs = [{"vpc_id": v["VpcId"], "cidr": v["CidrBlock"], "state": v["State"], "is_default": v["IsDefault"],
                     "name": next((t["Value"] for t in v.get("Tags", []) if t["Key"] == "Name"), "")}
                    for v in self.ec2.describe_vpcs()["Vpcs"]]
            return self.report("success", f"Found {len(vpcs)} VPCs", {"vpcs": vpcs})
        except Exception as e:
            return self.report("error", str(e))

    def diagnose_vpc(self, vpc_id: str) -> dict:
        try:
            f = [{"Name": "vpc-id", "Values": [vpc_id]}]
            subnets      = self.ec2.describe_subnets(Filters=f)["Subnets"]
            igws         = self.ec2.describe_internet_gateways(Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}])["InternetGateways"]
            route_tables = self.ec2.describe_route_tables(Filters=f)["RouteTables"]
            nat_gws      = self.ec2.describe_nat_gateways(Filter=f)["NatGateways"]
            public_subnets, private_subnets = [], []
            for rt in route_tables:
                has_igw = any(r.get("GatewayId", "").startswith("igw-") for r in rt["Routes"])
                for assoc in rt.get("Associations", []):
                    sid = assoc.get("SubnetId")
                    if sid:
                        (public_subnets if has_igw else private_subnets).append(sid)
            return self.report("success", f"VPC {vpc_id} diagnosed", {
                "vpc_id": vpc_id, "subnet_count": len(subnets),
                "has_internet_gateway": len(igws) > 0,
                "has_nat_gateway": any(n["State"] == "available" for n in nat_gws),
                "public_subnets": public_subnets, "private_subnets": private_subnets,
            })
        except Exception as e:
            return self.report("error", str(e))

    def list_subnets(self) -> dict:
        try:
            subnets = [{"subnet_id": s["SubnetId"], "vpc_id": s["VpcId"], "az": s["AvailabilityZone"],
                        "cidr": s["CidrBlock"], "available_ips": s["AvailableIpAddressCount"]}
                       for s in self.ec2.describe_subnets()["Subnets"]]
            return self.report("success", f"Found {len(subnets)} subnets", {"subnets": subnets})
        except Exception as e:
            return self.report("error", str(e))

    def list_security_groups(self) -> dict:
        try:
            sgs = [{"group_id": sg["GroupId"], "name": sg["GroupName"], "vpc_id": sg.get("VpcId", "")}
                   for sg in self.ec2.describe_security_groups()["SecurityGroups"]]
            return self.report("success", f"Found {len(sgs)} security groups", {"security_groups": sgs})
        except Exception as e:
            return self.report("error", str(e))

    def check_security_group_rules(self, group_id: str) -> dict:
        try:
            sg = self.ec2.describe_security_groups(GroupIds=[group_id])["SecurityGroups"][0]
            def fmt(rules):
                return [{"protocol": r.get("IpProtocol"), "from_port": r.get("FromPort", "all"),
                         "to_port": r.get("ToPort", "all"), "cidrs": [ip["CidrIp"] for ip in r.get("IpRanges", [])]} for r in rules]
            return self.report("success", f"SG {group_id} rules",
                               {"name": sg["GroupName"], "inbound": fmt(sg["IpPermissions"]), "outbound": fmt(sg["IpPermissionsEgress"])})
        except Exception as e:
            return self.report("error", str(e))

    def fix_security_group(self, task: dict) -> dict:
        try:
            self.ec2.authorize_security_group_ingress(GroupId=task["group_id"], IpPermissions=task["ip_permissions"])
            return self.report("success", f"Security group {task['group_id']} updated")
        except Exception as e:
            return self.report("error", str(e))
