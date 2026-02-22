-- Create notifications table
CREATE TABLE notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  type VARCHAR(20) NOT NULL CHECK (type IN ('SYSTEM', 'HEALTH_ALERT', 'REMINDER')),
  title VARCHAR(100) NOT NULL,
  message TEXT NOT NULL,
  channel VARCHAR(20) NOT NULL CHECK (channel IN ('IN_APP', 'EMAIL', 'PUSH')),
  is_read BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  read_at TIMESTAMPTZ
);

-- Index for user notifications
CREATE INDEX idx_notifications_user_id ON notifications(user_id);

-- Create notification preferences table
CREATE TABLE notification_preferences (
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  type VARCHAR(20) NOT NULL CHECK (type IN ('SYSTEM', 'HEALTH_ALERT', 'REMINDER')),
  channel VARCHAR(20) NOT NULL CHECK (channel IN ('IN_APP', 'EMAIL', 'PUSH')),
  enabled BOOLEAN NOT NULL DEFAULT true,
  PRIMARY KEY (user_id, type, channel)
);

-- Enable RLS
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_preferences ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only see their own notifications
CREATE POLICY user_notifications_isolation ON notifications
  USING (user_id = current_setting('app.current_user_id')::uuid);

-- RLS Policy: Users can only see their own preferences
CREATE POLICY user_preferences_isolation ON notification_preferences
  USING (user_id = current_setting('app.current_user_id')::uuid);
