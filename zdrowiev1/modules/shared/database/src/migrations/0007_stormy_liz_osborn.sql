CREATE TABLE IF NOT EXISTS "meal_products" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"meal_id" uuid NOT NULL,
	"name" varchar(255) NOT NULL,
	"barcode" varchar(50),
	"product_id" varchar(255),
	"quantity" integer NOT NULL,
	"calories" integer NOT NULL,
	"protein" integer NOT NULL,
	"carbs" integer NOT NULL,
	"fat" integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "meals" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"name" varchar(255) NOT NULL,
	"consumed_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "notification_preferences" (
	"user_id" uuid NOT NULL,
	"type" varchar(20) NOT NULL,
	"channel" varchar(20) NOT NULL,
	"enabled" boolean DEFAULT true NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "notifications" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"type" varchar(20) NOT NULL,
	"title" varchar(100) NOT NULL,
	"message" text NOT NULL,
	"channel" varchar(20) NOT NULL,
	"is_read" boolean DEFAULT false NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"read_at" timestamp
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "symptoms" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"report_id" uuid NOT NULL,
	"name" varchar(100) NOT NULL,
	"severity" integer NOT NULL,
	"duration_hours" integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "triage_results" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"report_id" uuid NOT NULL,
	"risk_level" varchar(20),
	"recommendation" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "activity_entries" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"date" date NOT NULL,
	"steps" integer DEFAULT 0 NOT NULL,
	"calories_burned" double precision DEFAULT 0 NOT NULL,
	"activity_type" varchar(50),
	"duration_minutes" integer,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
DROP TABLE "meal_entries";--> statement-breakpoint
DROP TABLE "diagnoses";--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "unit" text DEFAULT 'kg' NOT NULL;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "bmi" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "fat_percent" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "fat_kg" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "muscle_mass_kg" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "muscle_percent" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "water_percent" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "bmr_kcal" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "bone_mass_kg" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "protein_percent" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "lean_mass_kg" double precision;--> statement-breakpoint
ALTER TABLE "weight_readings" ADD COLUMN "metabolic_age" double precision;--> statement-breakpoint
ALTER TABLE "symptom_reports" DROP COLUMN IF EXISTS "description";--> statement-breakpoint
ALTER TABLE "symptom_reports" DROP COLUMN IF EXISTS "severity";--> statement-breakpoint
ALTER TABLE "symptom_reports" DROP COLUMN IF EXISTS "timestamp";--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "meal_products" ADD CONSTRAINT "meal_products_meal_id_meals_id_fk" FOREIGN KEY ("meal_id") REFERENCES "meals"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "meals" ADD CONSTRAINT "meals_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "notification_preferences" ADD CONSTRAINT "notification_preferences_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "notifications" ADD CONSTRAINT "notifications_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "symptoms" ADD CONSTRAINT "symptoms_report_id_symptom_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "symptom_reports"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "triage_results" ADD CONSTRAINT "triage_results_report_id_symptom_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "symptom_reports"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "activity_entries" ADD CONSTRAINT "activity_entries_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
