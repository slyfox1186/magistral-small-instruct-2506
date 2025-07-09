import { v4 as uuidv4 } from 'uuid';
import { createRef } from 'react';
import { Message } from '../api/chat';
import { Status } from '../types/status';

/**
 * Extended Message interface with UI-specific properties
 */
export interface AppMessage extends Message {
  id: string;
  nodeRef?: React.RefObject<HTMLDivElement>;
  status?: Status;
}

/**
 * Creates a new user message with proper formatting
 *
 * @param content The message content
 * @returns A properly formatted user message
 */
export function createUserMessage(content: string): AppMessage {
  return {
    id: uuidv4(),
    role: 'user',
    content,
    nodeRef: createRef<HTMLDivElement>(),
  };
}

/**
 * Creates a new assistant message with proper formatting
 *
 * @param content The message content
 * @returns A properly formatted assistant message
 */
export function createAssistantMessage(content: string, status?: Status): AppMessage {
  return {
    id: uuidv4(),
    role: 'assistant',
    content,
    nodeRef: createRef<HTMLDivElement>(),
    status,
  };
}

/**
 * Creates a new error message with proper formatting
 *
 * @param errorMsg The error message
 * @returns A properly formatted error message
 */
export function createErrorMessage(errorMsg: string): AppMessage {
  return {
    id: uuidv4(),
    role: 'assistant',
    content: errorMsg,
    nodeRef: createRef<HTMLDivElement>(),
  };
}

/**
 * Formats messages for sending to the backend API
 *
 * @param messages Array of AppMessages from chat history
 * @param userPrompt Current user prompt to add
 * @returns Array of Messages formatted for the API
 */
export function formatMessagesForAPI(messages: AppMessage[], userPrompt: string): Message[] {
  return [
    ...messages.map(({ role, content }) => ({ role, content })),
    { role: 'user', content: userPrompt },
  ];
}

export default {
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  formatMessagesForAPI,
};
