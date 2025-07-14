#!/usr/bin/env node
/**
 * Backend readiness checker for Vite
 * 
 * This script monitors port 8000 and only starts Vite when the backend is ready.
 * This prevents proxy errors and provides a clean startup experience.
 */

const net = require('net');
const { spawn } = require('child_process');

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8000;
const CHECK_INTERVAL = 1000; // 1 second
const MAX_WAIT_TIME = 120000; // 2 minutes
const STARTUP_GRACE_PERIOD = 2000; // 2 seconds after backend is ready

let startTime = Date.now();
let isBackendReady = false;
let viteProcess = null;

/**
 * Check if port is open and accepting connections
 */
function checkPort(host, port) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    
    const timeout = setTimeout(() => {
      socket.destroy();
      resolve(false);
    }, 3000); // 3 second timeout
    
    socket.connect(port, host, () => {
      clearTimeout(timeout);
      socket.destroy();
      resolve(true);
    });
    
    socket.on('error', () => {
      clearTimeout(timeout);
      resolve(false);
    });
  });
}

/**
 * Start Vite development server
 */
function startVite() {
  console.log('🚀 Starting Vite development server...');
  
  viteProcess = spawn('npm', ['run', 'dev'], {
    stdio: 'inherit',
    shell: true,
    env: {
      ...process.env,
      FORCE_COLOR: '1' // Preserve colors
    }
  });
  
  viteProcess.on('close', (code) => {
    console.log(`\n📦 Vite process exited with code ${code}`);
    process.exit(code);
  });
  
  viteProcess.on('error', (error) => {
    console.error('❌ Failed to start Vite:', error);
    process.exit(1);
  });
}

/**
 * Monitor backend readiness
 */
async function monitorBackend() {
  const elapsed = Date.now() - startTime;
  
  if (elapsed > MAX_WAIT_TIME) {
    console.error(`❌ Backend did not start within ${MAX_WAIT_TIME / 1000} seconds`);
    console.error('   Please check the backend logs and try again.');
    process.exit(1);
  }
  
  const isPortOpen = await checkPort(BACKEND_HOST, BACKEND_PORT);
  
  if (isPortOpen && !isBackendReady) {
    console.log('✅ Backend is responding on port 8000');
    console.log(`⏱️  Waiting ${STARTUP_GRACE_PERIOD / 1000} seconds for backend to fully initialize...`);
    
    isBackendReady = true;
    
    // Give the backend a moment to fully initialize before starting Vite
    setTimeout(() => {
      startVite();
    }, STARTUP_GRACE_PERIOD);
    
  } else if (!isPortOpen && !isBackendReady) {
    const dots = '.'.repeat((Math.floor(elapsed / 1000) % 3) + 1);
    process.stdout.write(`\r🔄 Waiting for backend to start on port 8000${dots}   `);
    
    setTimeout(monitorBackend, CHECK_INTERVAL);
  }
}

/**
 * Handle graceful shutdown
 */
function handleShutdown() {
  console.log('\n🛑 Shutting down...');
  
  if (viteProcess) {
    viteProcess.kill('SIGTERM');
    
    // Force kill after 5 seconds
    setTimeout(() => {
      if (viteProcess && !viteProcess.killed) {
        viteProcess.kill('SIGKILL');
      }
    }, 5000);
  }
  
  process.exit(0);
}

// Handle shutdown signals
process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);
process.on('SIGHUP', handleShutdown);

// Start monitoring
console.log('🔍 Checking for backend readiness...');
console.log(`   Backend: ${BACKEND_HOST}:${BACKEND_PORT}`);
console.log(`   Timeout: ${MAX_WAIT_TIME / 1000} seconds\n`);

monitorBackend();