-- ============================================================
-- infra/postgres/init.sql
-- Run once by the postgres container on first start.
-- ============================================================

-- Enforce UTC for all connections
SET timezone = 'UTC';

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Healthcheck role (read-only, no password needed inside container)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'healthcheck') THEN
    CREATE ROLE healthcheck LOGIN PASSWORD 'healthcheck_ro';
  END IF;
END
$$;
