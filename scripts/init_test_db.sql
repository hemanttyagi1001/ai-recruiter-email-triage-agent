-- Runs once, the first time the Postgres container is created.
-- Creates the sibling test database that TEST_DATABASE_URL points at.
CREATE DATABASE triage_test OWNER triage;
