DO $$
DECLARE
  t_name text;
  table_exists boolean;
BEGIN
  FOR t_name IN 
    SELECT unnest(ARRAY[
      'users',
      'weight_readings', 
      'heart_rate_readings', 
      'sleep_records', 
      'meal_entries', 
      'symptom_reports', 
      'notifications',
      'activity'
    ])
  LOOP
    SELECT EXISTS (
      SELECT FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_name = t_name::name
    ) INTO table_exists;

    IF table_exists THEN
      EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY;', t_name);
      
      IF t_name = 'users' THEN
        EXECUTE '
          CREATE POLICY "Users can only read their own data" ON "users" FOR SELECT USING (id = current_setting(''app.current_user_id'', true)::uuid);
          CREATE POLICY "Users can only update their own data" ON "users" FOR UPDATE USING (id = current_setting(''app.current_user_id'', true)::uuid);
        ';
      ELSE
        EXECUTE format('
          CREATE POLICY "Users can only select their own %I" ON "%I" FOR SELECT USING (user_id = current_setting(''app.current_user_id'', true)::uuid);
          CREATE POLICY "Users can only insert their own %I" ON "%I" FOR INSERT WITH CHECK (user_id = current_setting(''app.current_user_id'', true)::uuid);
          CREATE POLICY "Users can only update their own %I" ON "%I" FOR UPDATE USING (user_id = current_setting(''app.current_user_id'', true)::uuid);
          CREATE POLICY "Users can only delete their own %I" ON "%I" FOR DELETE USING (user_id = current_setting(''app.current_user_id'', true)::uuid);
        ', t_name, t_name, t_name, t_name, t_name, t_name, t_name, t_name);
      END IF;
      
    END IF;
  END LOOP;
END;
$$;
