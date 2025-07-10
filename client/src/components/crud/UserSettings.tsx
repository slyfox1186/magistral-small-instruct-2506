import React, { useState, useEffect, useCallback } from 'react';
import { UserSettings, UserSettingsUpdate, Theme } from '@/utils/types';
import { crudApi } from '@/api/crud';
import { useAlerts } from '@/hooks/useAlerts';
import { useTheme } from '@/hooks/useTheme';
import { THEMES } from '@/contexts/ThemeContextDefinition';

interface UserSettingsProps {
  userId: string;
}

export const UserSettingsComponent: React.FC<UserSettingsProps> = ({ userId }) => {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<UserSettingsUpdate>({});
  const [newPrompt, setNewPrompt] = useState('');
  const { showAlert } = useAlerts();
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
        custom_prompts: [...userSettings.custom_prompts],
      });
      setError(null);
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
            custom_prompts: [...defaultSettings.custom_prompts],
          });
        } catch {
          setError('Failed to create user settings');
          showAlert('Failed to load user settings', 'error');
        }
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load settings');
        showAlert('Failed to load user settings', 'error');
      }
    } finally {
      setLoading(false);
    }
  }, [userId, showAlert]);

  useEffect(() => {
    loadSettings();
  }, [userId, loadSettings]);

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
      
      showAlert('Settings saved successfully', 'success');
    } catch {
      showAlert('Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const handleAddPrompt = () => {
    if (!newPrompt.trim()) {
      showAlert('Please enter a prompt', 'warning');
      return;
    }

    const updatedPrompts = [...(formData.custom_prompts || []), newPrompt.trim()];
    setFormData((prev: UserSettingsUpdate) => ({ ...prev, custom_prompts: updatedPrompts }));
    setNewPrompt('');
  };

  const handleRemovePrompt = (index: number) => {
    const updatedPrompts = (formData.custom_prompts || []).filter((_: string, i: number) => i !== index);
    setFormData((prev: UserSettingsUpdate) => ({ ...prev, custom_prompts: updatedPrompts }));
  };

  const handleInputChange = (field: keyof UserSettingsUpdate, value: unknown) => {
    setFormData((prev: UserSettingsUpdate) => ({ ...prev, [field]: value }));
  };

  if (loading) {
    return (
      <div className="user-settings loading">
        <div className="loading-spinner">Loading settings...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="user-settings error">
        <div className="error-message">
          {error}
          <button onClick={loadSettings} className="btn btn-sm">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="user-settings">
      <div className="settings-header">
        <h2>User Settings</h2>
        <button
          onClick={handleSave}
          disabled={saving}
          className="btn btn-primary"
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>

      <div className="settings-form">
        <div className="settings-section">
          <h3>Appearance</h3>
          <div className="form-group">
            <label htmlFor="theme">Theme</label>
            <select
              id="theme"
              value={formData.theme || 'celestial-indigo'}
              onChange={(e) => handleInputChange('theme', e.target.value as Theme)}
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
          <h3>AI Behavior</h3>
          <div className="form-group">
            <label htmlFor="ai_personality">AI Personality</label>
            <select
              id="ai_personality"
              value={formData.ai_personality || 'helpful'}
              onChange={(e) => handleInputChange('ai_personality', e.target.value)}
            >
              <option value="helpful">Helpful</option>
              <option value="creative">Creative</option>
              <option value="analytical">Analytical</option>
              <option value="friendly">Friendly</option>
              <option value="professional">Professional</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="response_style">Response Style</label>
            <select
              id="response_style"
              value={formData.response_style || 'balanced'}
              onChange={(e) => handleInputChange('response_style', e.target.value)}
            >
              <option value="concise">Concise</option>
              <option value="balanced">Balanced</option>
              <option value="detailed">Detailed</option>
              <option value="technical">Technical</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="preferred_language">Preferred Language</label>
            <select
              id="preferred_language"
              value={formData.preferred_language || 'en'}
              onChange={(e) => handleInputChange('preferred_language', e.target.value)}
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
          <h3>Memory & Learning</h3>
          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.memory_retention ?? true}
                onChange={(e) => handleInputChange('memory_retention', e.target.checked)}
              />
              Enable Memory Retention
              <span className="help-text">Allow the AI to remember past conversations</span>
            </label>
          </div>

          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.auto_summarize ?? true}
                onChange={(e) => handleInputChange('auto_summarize', e.target.checked)}
              />
              Auto-summarize Long Conversations
              <span className="help-text">Automatically create summaries of lengthy chats</span>
            </label>
          </div>
        </div>

        <div className="settings-section">
          <h3>Custom Prompts</h3>
          <div className="custom-prompts">
            <div className="prompt-input">
              <input
                type="text"
                placeholder="Add a custom prompt..."
                value={newPrompt}
                onChange={(e) => setNewPrompt(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddPrompt()}
              />
              <button onClick={handleAddPrompt} className="btn btn-sm btn-primary">
                Add
              </button>
            </div>

            <div className="prompt-list">
              {(formData.custom_prompts || []).map((prompt: string, index: number) => (
                <div key={index} className="prompt-item">
                  <span className="prompt-text">{prompt}</span>
                  <button
                    onClick={() => handleRemovePrompt(index)}
                    className="btn btn-sm btn-danger"
                    title="Remove prompt"
                  >
                    ×
                  </button>
                </div>
              ))}
              {(!formData.custom_prompts || formData.custom_prompts.length === 0) && (
                <div className="empty-prompts">
                  No custom prompts yet. Add some to quickly start conversations!
                </div>
              )}
            </div>
          </div>
        </div>

        {settings && (
          <div className="settings-section">
            <h3>Account Information</h3>
            <div className="info-group">
              <div className="info-item">
                <label>User ID:</label>
                <span>{settings.user_id}</span>
              </div>
              <div className="info-item">
                <label>Created:</label>
                <span>{new Date(settings.created_at).toLocaleDateString()}</span>
              </div>
              <div className="info-item">
                <label>Last Updated:</label>
                <span>{new Date(settings.updated_at).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};