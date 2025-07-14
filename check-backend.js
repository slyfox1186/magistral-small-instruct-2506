#!/usr/bin/env node
/**
 * Backend Health Check Utility
 * 
 * Quick utility to check if the backend is running and healthy.
 */

const net = require('net');

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8000;

/**
 * Check if port is open
 */
function checkPort(host, port) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    
    const timeout = setTimeout(() => {
      socket.destroy();
      resolve(false);
    }, 3000);
    
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
 * Check backend health endpoint
 */
async function checkBackendHealth() {
  try {
    const response = await fetch('http://localhost:8000/api/health', {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    
    if (response.ok) {
      const data = await response.json();
      return { healthy: true, data };
    } else {
      return { healthy: false, error: `HTTP ${response.status}` };
    }
  } catch (error) {
    return { healthy: false, error: error.message };
  }
}

/**
 * Main check function
 */
async function main() {
  console.log('🔍 Checking backend status...\n');
  
  // Check if port is open
  console.log(`   Port ${BACKEND_PORT}: `, end='');
  const portOpen = await checkPort(BACKEND_HOST, BACKEND_PORT);
  
  if (portOpen) {
    console.log('✅ Open');
    
    // Check health endpoint
    console.log('   Health check: ', end='');
    const health = await checkBackendHealth();
    
    if (health.healthy) {
      console.log('✅ Healthy');
      console.log('\n✨ Backend is running and ready!');
      
      if (health.data) {
        console.log('\nHealth data:');
        console.log(JSON.stringify(health.data, null, 2));
      }
      
      process.exit(0);
    } else {
      console.log(`❌ Unhealthy (${health.error})`);
      console.log('\n⚠️  Backend is running but not healthy');
      process.exit(1);
    }
  } else {
    console.log('❌ Closed');
    console.log('\n❌ Backend is not running');
    console.log('   Start it with: python start.py');
    process.exit(1);
  }
}

// Polyfill for older Node.js versions
if (typeof fetch === 'undefined') {
  global.fetch = async (url, options = {}) => {
    const { default: fetch } = await import('node-fetch');
    return fetch(url, options);
  };
}

main().catch((error) => {
  console.error('❌ Health check failed:', error);
  process.exit(1);
});