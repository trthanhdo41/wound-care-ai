-- Migration: Add image_path column to messages table
-- Run this if your database already exists

ALTER TABLE messages ADD COLUMN image_path VARCHAR(500) NULL;
