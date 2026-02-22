-- Create activity_entries table
CREATE TABLE IF NOT EXISTS "activity_entries" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
	"date" date NOT NULL,
	"steps" integer DEFAULT 0 NOT NULL,
	"calories_burned" double precision DEFAULT 0 NOT NULL,
	"activity_type" varchar(50),
	"duration_minutes" integer,
	"created_at" timestamp DEFAULT now() NOT NULL
);

-- Index for user activity
CREATE INDEX IF NOT EXISTS idx_activity_entries_user_id ON activity_entries(user_id);

-- Enable RLS
ALTER TABLE "activity_entries" ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only see their own activity
DO $$ BEGIN
    CREATE POLICY user_activity_isolation ON "activity_entries"
      USING (user_id = current_setting('app.current_user_id', true)::uuid);
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;
