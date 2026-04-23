provider "aws" { region = var.aws_region }

resource "aws_ecs_cluster" "chemai" { name = "chemai-nexus" }

resource "aws_ecs_task_definition" "app" {
  family                   = "chemai-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  container_definitions    = jsonencode([{
    name  = "api", image = "${var.ecr_repo_url}:latest", cpu = 512, memory = 1024,
    portMappings = [{ containerPort = 8000, hostPort = 8000, protocol = "tcp" }],
    environment = [{ name = "DATABASE_URL", value = "postgresql://${var.db_user}:${var.db_pass}@${aws_rds_cluster.main.endpoint}:5432/chemai" }]
  }])
}

resource "aws_ecs_service" "api" {
  name            = "chemai-api"
  cluster         = aws_ecs_cluster.chemai.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.public_subnet_ids
    security_groups = [aws_security_group.api.id]
    assign_public_ip = true
  }
}

resource "aws_rds_cluster" "main" {
  cluster_identifier      = "chemai-db"
  engine                  = "aurora-postgresql"
  engine_version          = "15.5"
  database_name           = "chemai"
  master_username         = var.db_user
  master_password         = var.db_pass
  skip_final_snapshot     = true
}
