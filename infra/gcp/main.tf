provider "google" { project = var.gcp_project; region = var.gcp_region }

resource "google_cloud_run_v2_service" "api" {
  name     = "chemai-api"
  location = var.gcp_region
  template {
    containers {
      image = "${var.gcp_artifact_repo}/chemai:latest"
      ports { container_port = 8000 }
      env { name = "DATABASE_URL"; value = "postgresql+pg8000://${var.db_user}:${var.db_pass}@/chemai?unix_socket=/cloudsql/${google_sql_database_instance.main.connection_name}" }
    }
    vpc_access { connector = google_vpc_access_connector.chemai.id; egress = "ALL_TRAFFIC" }
  }
}

resource "google_sql_database_instance" "main" {
  name             = "chemai-db"
  database_version = "POSTGRES_15"
  region           = var.gcp_region
  settings {
    tier = "db-f1-micro"
    ip_configuration { ipv4_enabled = true }
  }
  deletion_protection = false
}
