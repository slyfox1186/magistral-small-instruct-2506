import React, { useState, useEffect, useCallback } from 'react';
import { UserSettings, UserSettingsUpdate, Theme } from '@/utils/types';
import { crudApi } from '@/api/crud';
import { useAlert } from '@/hooks/useAlerts';
import { useTheme } from '@/hooks/useTheme';
import { THEMES } from '@/contexts/ThemeContextDefinition';

interface ChatSettingsProps {
  userId: string;
  isOpen: boolean;
  onClose: () => void;
}

export const ChatSettings: React.FC<ChatSettingsProps> = ({ userId, isOpen, onClose }) => {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState<UserSettingsUpdate>({});
  const alert = useAlert();
  const { setTheme } = useTheme();

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      const userSettings = await crudApi.getUserSettings(userId);
      setSettings(userSettings);
      setFormData({
        theme: userSettings.theme,
        ai_personality: userSettings.ai_personality,
        response_style: userSettings.response_style,
        memory_retention: userSettings.memory_retention,
        auto_summarize: userSettings.auto_summarize,
        preferred_language: userSettings.preferred_language,
      });
    } catch (err) {
      // If settings don't exist, create default ones
      if (err instanceof Error && err.message.includes('404')) {
        try {
          const defaultSettings = await crudApi.createUserSettings(userId, {
            user_id: userId,
            theme: 'celestial-indigo',
            ai_personality: 'helpful',
            response_style: 'balanced',
            memory_retention: true,
            auto_summarize: true,
            preferred_language: 'en',
            custom_prompts: [],
          });
          setSettings(defaultSettings);
          setFormData({
            theme: defaultSettings.theme,
            ai_personality: defaultSettings.ai_personality,
            response_style: defaultSettings.response_style,
            memory_retention: defaultSettings.memory_retention,
            auto_summarize: defaultSettings.auto_summarize,
            preferred_language: defaultSettings.preferred_language,
          });
        } catch {
          alert.error('Failed to create user settings');
        }
      } else {
        alert.error('Failed to load user settings');
      }
    } finally {
      setLoading(false);
    }
  }, [userId]); // Removed alert dependency to prevent infinite loop

  useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen, loadSettings]);

  const handleSave = async () => {
    if (!settings) return;

    try {
      setSaving(true);
      const updated = await crudApi.updateUserSettings(userId, formData);
      setSettings(updated);
      
      // Update theme immediately if changed
      if (formData.theme && formData.theme !== settings.theme) {
        setTheme(formData.theme);
      }
      
      alert.success('Settings saved successfully');
      onClose();
    } catch {
      alert.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleInputChange = (field: keyof UserSettingsUpdate, value: unknown) => {
    setFormData((prev: UserSettingsUpdate) => ({ ...prev, [field]: value }));
  };

  if (!isOpen) return null;

  return (
    <div className="chat-settings-overlay">
      <div className="chat-settings-panel">
        <div className="settings-header">
          <h3>Chat Settings</h3>
          <button onClick={onClose} className="close-button" aria-label="Close settings">
            ×
          </button>
        </div>

        {loading ? (
          <div className="settings-loading">
            <div className="loading-spinner"></div>
            <span>Loading settings...</span>
          </div>
        ) : (
          <div className="settings-content">
            <div className="settings-section">
              <h4>Appearance</h4>
              <div className="form-group">
                <label htmlFor="chat-theme">Theme</label>
                <select
                  id="chat-theme"
                  value={formData.theme || 'celestial-indigo'}
                  onChange={(e) => handleInputChange('theme', e.target.value as Theme)}
                  className="settings-select"
                >
                  {THEMES.map((themeName) => (
                    <option key={themeName} value={themeName}>
                      {themeName.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="settings-section">
              <h4>AI Behavior</h4>
              <div className="form-group">
                <label htmlFor="chat-personality">AI Personality</label>
                <select
                  id="chat-personality"
                  value={formData.ai_personality || 'helpful'}
                  onChange={(e) => handleInputChange('ai_personality', e.target.value)}
                  className="settings-select"
                >
                  <option value="helpful">Helpful</option>
                  <option value="creative">Creative</option>
                  <option value="analytical">Analytical</option>
                  <option value="friendly">Friendly</option>
                  <option value="professional">Professional</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="chat-response-style">Response Style</label>
                <select
                  id="chat-response-style"
                  value={formData.response_style || 'balanced'}
                  onChange={(e) => handleInputChange('response_style', e.target.value)}
                  className="settings-select"
                >
                  <option value="concise">Concise</option>
                  <option value="balanced">Balanced</option>
                  <option value="detailed">Detailed</option>
                  <option value="technical">Technical</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="chat-language">Preferred Language</label>
                <select
                  id="chat-language"
                  value={formData.preferred_language || 'en'}
                  onChange={(e) => handleInputChange('preferred_language', e.target.value)}
                  className="settings-select"
                >
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                  <option value="it">Italian</option>
                  <option value="pt">Portuguese</option>
                  <option value="zh">Chinese</option>
                  <option value="ja">Japanese</option>
                </select>
              </div>
            </div>

            <div className="settings-section">
              <h4>Memory & Learning</h4>
              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={formData.memory_retention ?? true}
                    onChange={(e) => handleInputChange('memory_retention', e.target.checked)}
                  />
                  <span className="checkbox-text">
                    Enable Memory Retention
                    <small>Allow the AI to remember past conversations</small>
                  </span>
                </label>
              </div>

              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={formData.auto_summarize ?? true}
                    onChange={(e) => handleInputChange('auto_summarize', e.target.checked)}
                  />
                  <span className="checkbox-text">
                    Auto-summarize Long Conversations
                    <small>Automatically create summaries of lengthy chats</small>
                  </span>
                </label>
              </div>
            </div>

            <div className="settings-actions">
              <button onClick={onClose} className="btn btn-secondary">
                Cancel
              </button>
              <button 
                onClick={handleSave} 
                disabled={saving}
                className="btn btn-primary"
              >
                {saving ? 'Saving...' : 'Save Settings'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatSettings;