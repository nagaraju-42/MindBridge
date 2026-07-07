-- ============================================================
-- MindBridge Supabase Schema
-- Anurag University CSE Mini-Project 2026-27
-- Run ALL of this in Supabase Dashboard → SQL Editor
-- ============================================================

-- Step 1: Enable the uuid-ossp extension (needed for gen_random_uuid)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TABLE 1: submissions
-- Stores anonymous student wellness check results.
-- NO PII (no name, no roll number, no email, no phone).
-- Only an auto-generated UUID identifies each submission.
-- ============================================================
CREATE TABLE IF NOT EXISTS submissions (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Routing info (non-identifying)
    department          TEXT,
    mode                TEXT        CHECK (mode IN ('text_analyser', 'form')),

    -- Form responses (10 questions — radio values, not PII)
    q1                  TEXT,
    q2                  TEXT,
    q3                  TEXT,
    q4                  TEXT,
    q5                  TEXT,
    q6                  TEXT,
    q7                  TEXT,
    q8                  TEXT,
    q9                  TEXT,
    q10_open            TEXT,

    -- Generated text from form (first 500 chars)
    generated_text      TEXT,

    -- ML Prediction output
    prediction          INTEGER     CHECK (prediction IN (0, 1)),
    confidence          FLOAT       CHECK (confidence >= 0.0 AND confidence <= 1.0),
    risk_level          TEXT        CHECK (risk_level IN ('low', 'medium', 'high')),
    shap_top_words      TEXT,

    -- Counselor connect tracking
    whatsapp_connected  BOOLEAN     DEFAULT FALSE
);

-- Index for fast counselor dashboard queries
CREATE INDEX IF NOT EXISTS idx_submissions_created_at  ON submissions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_risk_level  ON submissions (risk_level);
CREATE INDEX IF NOT EXISTS idx_submissions_department  ON submissions (department);
CREATE INDEX IF NOT EXISTS idx_submissions_mode        ON submissions (mode);

-- ============================================================
-- TABLE 2: campaigns
-- Tracks email campaigns sent by the counselor.
-- ============================================================
CREATE TABLE IF NOT EXISTS campaigns (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    subject         TEXT        NOT NULL,
    body            TEXT        NOT NULL,
    department      TEXT,
    sent_count      INTEGER     DEFAULT 0,
    status          TEXT        DEFAULT 'sent' CHECK (status IN ('draft', 'sent', 'failed')),
    sender_note     TEXT
);

CREATE INDEX IF NOT EXISTS idx_campaigns_created_at ON campaigns (created_at DESC);

-- ============================================================
-- TABLE 3: counselors
-- Optional: store counselor contact info in DB.
-- Can also be managed via secrets.toml config.
-- ============================================================
CREATE TABLE IF NOT EXISTS counselors (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT        NOT NULL,
    department      TEXT,
    wa_number       TEXT,       -- 91XXXXXXXXXX format
    email           TEXT,
    calendly_url    TEXT,
    active          BOOLEAN     DEFAULT TRUE
);

-- ============================================================
-- Row Level Security (RLS)
-- Enable to restrict anon key to INSERT only on submissions.
-- Counselor dashboard uses service_role key to bypass RLS.
-- ============================================================
ALTER TABLE submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE campaigns   ENABLE ROW LEVEL SECURITY;
ALTER TABLE counselors  ENABLE ROW LEVEL SECURITY;

-- Allow anonymous users to INSERT submissions (students submitting forms)
CREATE POLICY "anon_insert_submissions"
    ON submissions FOR INSERT
    TO anon
    WITH CHECK (true);

-- Allow anonymous users to SELECT their own submission by id
-- (for the token lookup — student looks up their own token)
CREATE POLICY "anon_select_own_submission"
    ON submissions FOR SELECT
    TO anon
    USING (true);   -- open read for now; restrict to id= in production

-- Service role (used by dashboard) has full access — no policy needed
-- because service_role bypasses RLS automatically.

-- ============================================================
-- Seed: Insert a test counselor row
-- ============================================================
INSERT INTO counselors (name, department, wa_number, email, active)
VALUES
    ('Dr. Sample Counselor 1', 'CSE / IT / AI / DS', '919999999901', 'counselor1@anurag.edu.in', TRUE),
    ('Dr. Sample Counselor 2', 'ECE / EEE',          '919999999902', 'counselor2@anurag.edu.in', TRUE),
    ('Dr. Sample Counselor 3', 'Mech / Civil / Other','919999999903', 'counselor3@anurag.edu.in', TRUE)
ON CONFLICT DO NOTHING;

-- ============================================================
-- Verify: Run these SELECTs to confirm tables created
-- ============================================================
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
-- Expected output: campaigns, counselors, submissions
