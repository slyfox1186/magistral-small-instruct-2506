#!/usr/bin/env node
/**
 * Coordinated Startup Script
 * 
 * This script provides intelligent startup coordination:
 * 1. Starts the backend first
 * 2. Waits for backend to be ready on port 8000
 * 3. Then starts the frontend with Vite
 * 
 * This eliminates proxy errors and provides a smooth development experience.
 */

const { spawn } = require('child_process');
const net = require('net');
const path = require('path');

// Get current directory in a modern way
const currentDir = process.cwd();

const BACKEND_PORT = 8000;
const CHECK_INTERVAL = 1000; // 1 second
const BACKEND_STARTUP_TIMEOUT = 180000; // 3 minutes for model loading
const HEALTH_CHECK_RETRIES = 3;

let backendProcess = null;
let frontendProcess = null;
let isShuttingDown = false;

/**
 * Check if port is accepting connections
 */
function checkPort(host, port, timeout = 3000) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    
    const timer = setTimeout(() => {
      socket.destroy();
      resolve(false);
    }, timeout);
    
    socket.connect(port, host, () => {
      clearTimeout(timer);
      socket.destroy();
      resolve(true);
    });
    
    socket.on('error', () => {
      clearTimeout(timer);
      resolve(false);
    });
  });
}

/**
 * Check backend health endpoint
 */
async function checkBackendHealth() {
  try {
    const response = await fetch('http://localhost:8000/api/health', {
      method: 'GET',
      timeout: 5000
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Wait for backend to be fully ready
 */
async function waitForBackend() {
  const startTime = Date.now();
  let lastStatus = '';
  
  console.log('🔄 Waiting for backend to start...');
  console.log('   This may take 2-5 minutes while the LLM model loads');
  
  while (Date.now() - startTime < BACKEND_STARTUP_TIMEOUT) {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const status = `   ⏱️  ${elapsed}s elapsed`;
    
    if (status !== lastStatus) {
      process.stdout.write(`\r${status}   `);
      lastStatus = status;
    }
    
    // First check if port is open
    const portOpen = await checkPort('127.0.0.1', BACKEND_PORT, 2000);
    
    if (portOpen) {
      console.log('\n✅ Backend port is open, checking health...');
      
      // Then verify health endpoint responds
      let healthRetries = HEALTH_CHECK_RETRIES;
      while (healthRetries > 0) {
        const isHealthy = await checkBackendHealth();
        if (isHealthy) {
          console.log('✅ Backend is healthy and ready!');
          return true;
        }
        
        console.log(`   🔄 Health check failed, retrying... (${healthRetries} attempts left)`);
        healthRetries--;
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      
      console.log('⚠️  Backend port is open but health check failing, continuing anyway...');
      return true;
    }
    
    await new Promise(resolve => setTimeout(resolve, CHECK_INTERVAL));
  }
  
  return false;
}

/**
 * Start backend process
 */
function startBackend() {
  console.log('🚀 Starting backend...');
  
  backendProcess = spawn('python', ['start.py'], {
    cwd: currentDir,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1'
    }
  });
  
  backendProcess.stdout.on('data', (data) => {
    const output = data.toString();
    // Only show important backend messages
    if (output.includes('ERROR') || output.includes('CRITICAL') || output.includes('✅') || output.includes('🚀')) {
      process.stdout.write(`[BACKEND] ${output}`);
    }
  });
  
  backendProcess.stderr.on('data', (data) => {
    const output = data.toString();
    // Show all stderr output as it's likely important
    process.stdout.write(`[BACKEND] ${output}`);
  });
  
  backendProcess.on('close', (code) => {
    if (!isShuttingDown) {
      console.log(`\n❌ Backend process exited with code ${code}`);
      process.exit(code);
    }
  });
  
  backendProcess.on('error', (error) => {
    console.error('❌ Failed to start backend:', error);
    process.exit(1);
  });
}

/**
 * Start frontend process
 */
function startFrontend() {
  console.log('🎨 Starting frontend...');
  
  frontendProcess = spawn('npm', ['run', 'dev'], {
    cwd: path.join(currentDir, 'client'),
    stdio: 'inherit',
    shell: true,
    env: {
      ...process.env,
      FORCE_COLOR: '1'
    }
  });
  
  frontendProcess.on('close', (code) => {
    if (!isShuttingDown) {
      console.log(`\n📦 Frontend process exited with code ${code}`);
      process.exit(code);
    }
  });
  
  frontendProcess.on('error', (error) => {
    console.error('❌ Failed to start frontend:', error);
    process.exit(1);
  });
}

/**
 * Graceful shutdown
 */
function handleShutdown() {
  if (isShuttingDown) return;
  isShuttingDown = true;
  
  console.log('\n🛑 Shutting down servers...');
  
  if (frontendProcess) {
    frontendProcess.kill('SIGTERM');
  }
  
  if (backendProcess) {
    backendProcess.kill('SIGTERM');
  }
  
  // Force kill after 10 seconds
  setTimeout(() => {
    if (frontendProcess && !frontendProcess.killed) {
      frontendProcess.kill('SIGKILL');
    }
    if (backendProcess && !backendProcess.killed) {
      backendProcess.kill('SIGKILL');
    }
    process.exit(0);
  }, 10000);
}

// Handle shutdown signals
process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);
process.on('SIGHUP', handleShutdown);

/**
 * Main startup sequence
 */
async function main() {
  console.log('🎯 Neural Consciousness Chat System - Coordinated Startup');
  console.log('============================================================\n');
  
  try {
    // Step 1: Start backend
    startBackend();
    
    // Step 2: Wait for backend to be ready
    const backendReady = await waitForBackend();
    
    if (!backendReady) {
      console.error(`❌ Backend failed to start within ${BACKEND_STARTUP_TIMEOUT / 1000} seconds`);
      console.error('   Check the backend logs and try again.');
      process.exit(1);
    }
    
    // Step 3: Start frontend
    console.log('\n🎨 Backend is ready! Starting frontend...\n');
    startFrontend();
    
    console.log('\n✨ All services are running!');
    console.log('   Frontend: http://localhost:4000');
    console.log('   Backend:  http://localhost:8000');
    console.log('\n   Press Ctrl+C to stop all services\n');
    
  } catch (error) {
    console.error('❌ Startup failed:', error);
    process.exit(1);
  }
}

// Polyfill fetch for Node.js versions that don't have it
if (typeof fetch === 'undefined') {
  global.fetch = async (url, options = {}) => {
    const { default: fetch } = await import('node-fetch');
    return fetch(url, options);
  };
}

main();