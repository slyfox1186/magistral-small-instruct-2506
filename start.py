#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import signal
import time
import argparse
import traceback
import socket
import http.client
from datetime import datetime
import io

# Default ports
DEFAULT_FRONTEND_PORT = 4000
DEFAULT_BACKEND_PORT = 8000

# Process management timeouts
GRACEFUL_SHUTDOWN_TIMEOUT = 5.0  # 5 seconds for graceful shutdown
THREAD_JOIN_TIMEOUT = 2.0  # 2 seconds for thread joining

# Log file path
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "backend.log")

# ANSI color codes for better readability - improved color grouping
COLORS = {
    "RESET": "\033[0m",
    # Process identifiers
    "FRONTEND": "\033[96m",   # Cyan
    "BACKEND": "\033[92m",    # Green
    "SCRIPT": "\033[95m",     # Magenta
    
    # Log levels
    "INFO": "\033[94m",       # Blue
    "DEBUG": "\033[37m",      # Light gray
    "WARNING": "\033[93m",    # Yellow
    "ERROR": "\033[91m",      # Red
    "CRITICAL": "\033[97;41m", # White on red background
    
    # Special categories
    "SUCCESS": "\033[92m",    # Green
    "TRACE": "\033[90m",      # Dark gray
    
    # Model specific categories
    "MODEL_INFO": "\033[36m",  # Light Cyan
    "MODEL_DEBUG": "\033[34m", # Blue
    "LOCK_DEBUG": "\033[36m",  # Light Cyan
    "MEMORY_OPS": "\033[35m",  # Purple
    "REDIS_INFO": "\033[94m",  # Blue
}

running_processes = []
verbose_mode = True  # ALWAYS verbose for debugging
log_file = None
shutdown_performed = False  # Track if shutdown has already been performed

def log_message(prefix, message, color_key="INFO"):
    """Log a message with timestamp and prefix and also write to log file"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    final_color_key = color_key
    # If the default/generic "INFO" key is provided (or was the default),
    # try to use a color specific to the prefix if that prefix is defined in COLORS.
    if color_key == "INFO" and prefix in COLORS:
        final_color_key = prefix
    
    # Use the determined color key, with a fallback to the "INFO" color (blue) if the key is not found.
    color = COLORS.get(final_color_key, COLORS["INFO"])
    log_line = f"[{timestamp}] {prefix}: {message}"
    colored_log_line = f"{color}{log_line}{COLORS['RESET']}"
    
    # Write to console with color
    print(colored_log_line)
    
    # Write to log file without color codes
    if log_file and not log_file.closed:
        try:
            log_file.write(f"{log_line}\n")
            log_file.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}")

def log_debug(prefix, message):
    """Log a debug message if verbose mode is enabled"""
    if verbose_mode:
        log_message(prefix, message, "DEBUG")

def determine_log_level(text):
    """Determine the log level based on text content"""
    text_lower = text.lower()
    
    # MODEL-SPECIFIC CATEGORIZATION (should be checked before generic error patterns)
    
    # Lock-related debug information (not errors)
    if text.startswith("[LOCK DEBUG]") or "lock state" in text_lower or "lock held by" in text_lower:
        return "LOCK_DEBUG"
    
    # Memory extraction operations (not errors)
    if text.startswith("[MEMORY") or "memory_extraction" in text_lower or "memory extraction" in text_lower:
        return "MEMORY_OPS"
        
    # Model-related information logs and performance metrics (not errors)
    if text.startswith("llama_print:") or text.startswith("llama_model_loader:"):
        return "MODEL_INFO"
        
    if text.startswith("llama_ctx_tensor_print:") or "tensor_print" in text_lower:
        return "MODEL_DEBUG"
        
    # Load tensors messages - these are informational, not errors
    if text.startswith("load_tensors:") or "assigned to device" in text_lower:
        return "MODEL_INFO"
        
    # Llama context construction messages
    if text.startswith("llama_context:") or "constructing llama_context" in text_lower:
        return "MODEL_INFO"
        
    # CUDA-related loading messages
    if "cuda" in text_lower and any(term in text_lower for term in ["buffer", "model", "layer", "offload"]):
        return "MODEL_INFO"
        
    # Memory allocation and model loading messages
    if any(term in text_lower for term in ["create_memory", "set_abort_callback", "llama_kv_cache_unified"]):
        return "MODEL_INFO"
        
    # Any dotted lines or model initialization patterns
    if text.strip().startswith("..") or "constructing" in text_lower:
        return "MODEL_INFO"
        
    # Redis-related logs (not errors)
    if "redis_utils" in text_lower or "redis client" in text_lower or "redis." in text_lower:
        if "error" in text_lower and not "connection" in text_lower:
            return "ERROR"
        if "warning" in text_lower:
            return "WARNING"
        return "INFO"
        
    # Progress bars, batches, and metrics (not errors)
    if text.startswith("Llama.generate:") or text.startswith("Batches:") or "|" in text and ("%" in text or "it/s" in text):
        return "MODEL_INFO"
        
    # Various timing metrics (not errors)
    if any(term in text_lower for term in ["load time", "eval time", "prompt eval time", "total time", "completed in"]):
        return "MODEL_INFO"
    
    # Stream-related information (not errors)
    if text.startswith("llama_get_context_print:") or text.startswith("llama_perf_context_print:"):
        return "MODEL_INFO"
        
    # Prefix-match hit messages are informational, not errors
    if "prefix-match hit" in text_lower:
        return "MODEL_INFO"
        
    # STANDARD LOG LEVELS
    
    # Check for explicit log level indicators in FastAPI/Uvicorn logs
    if ("error:" in text_lower and not "finished attempt to send" in text_lower) or \
       ("exception" in text_lower and not "signal" in text_lower) or \
       ("fail" in text_lower and not "signal" in text_lower):
        return "ERROR"
    elif "warning:" in text_lower or "warn:" in text_lower:
        return "WARNING"
    elif text.startswith("INFO:"):
        return "INFO"
    elif text.startswith("DEBUG:"):
        return "DEBUG"
    
    # Special patterns for backend logs (with exclusions for false positives)
    if ("error" in text_lower and not any(term in text_lower for term in ["attempting to", "stream finished"])):
        return "ERROR"
    elif "warning" in text_lower:
        return "WARNING"
    elif "debug" in text_lower:
        return "DEBUG"
    elif "print_info:" in text_lower:
        return "INFO"
    elif "load_tensors:" in text_lower or "load time =" in text_lower:
        return "TRACE"
    
    # Special content-based coloring
    if any(success_term in text_lower for success_term in [
        "started server", "available", "success", "initialized", "loaded", "completed"
    ]):
        return "SUCCESS"
    elif any(warning_term in text_lower for warning_term in [
        "deprecated", "not recommended", "may not"
    ]):
        return "WARNING"
    
    # Model loading and initialization info (common in LLM backends)
    if any(model_term in text_lower for model_term in [
        "model", "n_embd", "layer", "cuda", "tokens", "tensor"
    ]):
        return "MODEL_INFO" 
    
    # Default based on prefix - most logs will be INFO level
    return "INFO"

def stream_output(process, prefix, default_color_key):
    """Stream process output with appropriate prefix and color based on content"""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            text = line.decode('utf-8').rstrip()
            
            # Determine color based on content for backend
            if prefix == "BACKEND":
                color_key = determine_log_level(text)
            else:
                color_key = default_color_key
                
            # Use log_message to handle both console and file output
            log_message(prefix, text, color_key)
    
    # Check for errors when the process ends
    return_code = process.poll()
    if return_code != 0:
        log_message(prefix, f"Process exited with code {return_code}", "ERROR")

def _read_and_log_stream(pipe, prefix, stream_name, get_color_key_func):
    """Helper function to read lines from a pipe and log them."""
    try:
        for line in pipe:
            line = line.strip()
            # Log ALL lines, even empty ones might be meaningful
            # Determine log level and color appropriately
            color_key = get_color_key_func(line)
            
            # For informational content from stderr, show as OUT instead of STDERR
            if stream_name == "STDERR" and color_key in ["MODEL_INFO", "MODEL_DEBUG", "LOCK_DEBUG", "MEMORY_OPS", "INFO", "SUCCESS", "TRACE"]:
                display_stream = "OUT"
            else:
                display_stream = stream_name
                
            log_message(prefix, f"{display_stream}: {line}", color_key)
    except Exception as e:
        # Log errors during stream reading, but don't crash the logging thread
        log_message(prefix, f"Error reading from {stream_name}: {str(e)}", "ERROR")
        log_debug(prefix, traceback.format_exc())
    finally:
        # Ensure the pipe is closed when reading is done or an error occurs
        if hasattr(pipe, 'close') and not pipe.closed:
            try:
                pipe.close()
            except Exception as e:
                log_message(prefix, f"Error closing {stream_name} pipe: {str(e)}", "ERROR")

def stream_process_output(process, prefix):
    """
    Stream and log output (stdout and stderr) from a process in real-time using separate threads.
    This function runs in a separate thread for each monitored process.
    """
    stdout_thread = None
    stderr_thread = None
    
    try:
        # Determine color logic for stdout with improved categorization
        def get_stdout_color_key(line_text):
            if prefix == "BACKEND":
                # Use the enhanced log level determination
                return determine_log_level(line_text)
            # Other processes use their own color or fall back to INFO
            return prefix if prefix in COLORS else "INFO"

        # Thread for stdout
        stdout_thread = threading.Thread(
            target=_read_and_log_stream,
            args=(process.stdout, prefix, "OUT", get_stdout_color_key),
            daemon=True
        )
        stdout_thread.start()

        # Thread for stderr
        stderr_thread = threading.Thread(
            target=_read_and_log_stream,
            args=(process.stderr, prefix, "STDERR", lambda line_text: determine_log_level(line_text)), # Use determine_log_level for stderr too
            daemon=True
        )
        stderr_thread.start()

        # Wait for the process to complete
        process.wait()

    except Exception as e:
        log_message(prefix, f"Error managing stream threads or process: {str(e)}", "ERROR")
        log_debug(prefix, traceback.format_exc())
    finally:
        # Ensure threads are joined (optional if daemon, but good practice for cleanup)
        join_thread_safely(stdout_thread, "stdout")
        join_thread_safely(stderr_thread, "stderr")

        # Log termination status after streams are (supposedly) flushed
        # process.poll() should be set now that process.wait() has returned
        return_code = process.poll()
        if return_code is not None:
            log_message(prefix, f"Process terminated with code {return_code}", "WARNING" if return_code != 0 else "INFO")
        else:
            # This case should ideally not be reached if process.wait() completed
            log_message(prefix, "Process termination status unknown after wait", "ERROR")

def cleanup(signum=None, frame=None):
    """Enhanced cleanup with aggressive port clearing for 4000 and 8000"""
    global shutdown_performed
    
    # Check if shutdown has already been performed
    if shutdown_performed:
        log_message("SCRIPT", "⚠️ Shutdown already performed, skipping duplicate cleanup", "INFO")
        return
    
    shutdown_performed = True
    log_message("SCRIPT", "🛑 INITIATING SHUTDOWN SEQUENCE...", "WARNING")
    
    # Step 1: Terminate our known processes gracefully first
    log_message("SCRIPT", "Terminating tracked processes...", "INFO")
    for process in running_processes:
        terminate_process_gracefully(process, GRACEFUL_SHUTDOWN_TIMEOUT)
    
    # Step 2: FORCE KILL everything on ports 4000 and 8000 (only if they are occupied)
    log_message("SCRIPT", "🔥 Checking critical ports...", "WARNING")
    critical_ports = [4000, 8000]
    
    for port in critical_ports:
        ensure_port_free(port)
    
    # Step 3: Additional cleanup for known server processes
    log_message("SCRIPT", "🧹 Cleaning up any remaining server processes...", "INFO")
    patterns = ['npm.*dev', 'vite.*4000', 'uvicorn.*8000', 'python.*main:app']
    kill_processes_by_patterns(patterns)
    
    # Step 4: Final verification
    log_message("SCRIPT", "🔍 Verifying ports are free...", "INFO")
    for port in critical_ports:
        if check_port_available(port):
            log_message("SCRIPT", f"✅ Port {port} confirmed free", "SUCCESS")
        else:
            log_message("SCRIPT", f"⚠️ Port {port} still occupied - manual intervention may be needed", "WARNING")
    
    log_message("SCRIPT", "✅ SHUTDOWN SEQUENCE COMPLETE", "SUCCESS")
    
    # Close log file if open
    if log_file and not log_file.closed:
        try:
            log_file.close()
            print(f"Closed log file: {LOG_FILE}")
        except Exception as e:
            print(f"Error closing log file: {e}")
    
    sys.exit(0)

def check_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def ensure_port_free(port, port_name=""):
    """
    Ensures a port is free. If occupied, attempts to free it.
    Returns True if port is free (or was successfully freed), False otherwise.
    """
    if check_port_available(port):
        log_message("SCRIPT", f"✅ {port_name}Port {port} is already free", "INFO")
        return True
    else:
        log_message("SCRIPT", f"⚠️ {port_name}Port {port} is occupied, clearing...", "WARNING")
        success = free_port(port)
        if success:
            log_message("SCRIPT", f"✅ {port_name}Port {port} successfully cleared", "SUCCESS")
        else:
            log_message("SCRIPT", f"❌ {port_name}Port {port} could not be cleared", "ERROR")
        return success

def ensure_port_free_with_retries(port, port_name="", max_attempts=5):
    """
    Ensures a port is free with multiple retry attempts.
    Returns True if successful, False if all attempts failed.
    """
    for attempt in range(max_attempts):
        if check_port_available(port):
            log_message("SCRIPT", f"✅ {port_name}Port {port} is free (attempt {attempt + 1})", "SUCCESS")
            return True
            
        log_message("SCRIPT", f"🔥 {port_name}Port {port} occupied - NUCLEAR ASSAULT ATTEMPT {attempt + 1}/{max_attempts}", "WARNING")
        
        # Try freeing the port
        free_port(port)
        
        # Wait a bit for cleanup
        time.sleep(1)
        
        # Check if we freed it
        if check_port_available(port):
            log_message("SCRIPT", f"✅ {port_name}Port {port} freed on attempt {attempt + 1}", "SUCCESS")
            return True
        elif attempt == max_attempts - 1:
            log_message("SCRIPT", f"💥 FAILED TO FREE {port_name}PORT {port} AFTER {max_attempts} NUCLEAR ATTEMPTS", "ERROR")
            log_message("SCRIPT", f"🚨 MANUAL INTERVENTION REQUIRED - Something is REALLY stuck on {port_name}port {port}", "ERROR")
            return False
        else:
            log_message("SCRIPT", f"💀 {port_name}Port {port} still occupied, escalating to next attempt...", "WARNING")
    
    return False

def terminate_process_gracefully(process, timeout=GRACEFUL_SHUTDOWN_TIMEOUT):
    """
    Terminates a process gracefully with proper timeout handling, then kills it if still running.
    """
    if process.poll() is None:  # If process is still running
        try:
            log_message("SCRIPT", f"Gracefully terminating process PID {process.pid} (timeout: {timeout}s)...", "INFO")
            process.terminate()
            
            # Wait for process to terminate gracefully
            try:
                process.wait(timeout=timeout)
                log_message("SCRIPT", f"Process PID {process.pid} terminated gracefully", "SUCCESS")
            except subprocess.TimeoutExpired:
                # Process didn't terminate within timeout, force kill it
                log_message("SCRIPT", f"Graceful termination timeout, force killing process PID {process.pid}...", "WARNING")
                process.kill()
                try:
                    process.wait(timeout=2.0)  # Give it 2 more seconds after kill
                    log_message("SCRIPT", f"Process PID {process.pid} force killed successfully", "WARNING")
                except subprocess.TimeoutExpired:
                    log_message("SCRIPT", f"Process PID {process.pid} could not be killed - may be zombie", "ERROR")
                    
        except Exception as e:
            log_message("SCRIPT", f"Error terminating process PID {process.pid}: {e}", "ERROR")

def join_thread_safely(thread, thread_name="", timeout=THREAD_JOIN_TIMEOUT):
    """
    Safely joins a thread with timeout and error handling.
    """
    if thread and thread.is_alive():
        try:
            thread.join(timeout=timeout)
        except Exception:
            pass  # Ignore errors during join

def kill_processes_by_patterns(patterns):
    """
    Kill processes matching the given patterns on Linux.
    patterns: list of shell patterns to match processes
    """
    try:
        for pattern in patterns:
            subprocess.run(f"pkill -f '{pattern}' 2>/dev/null || true", shell=True, timeout=3)
    except Exception as e:
        log_message("SCRIPT", f"Error in process cleanup: {e}", "WARNING")

def free_port(port):
    """
    ULTRA-AGGRESSIVE port freeing with multiple killing strategies.
    Will try every possible method to kill processes on the port.
    """
    log_message("SCRIPT", f"🔥 FORCE KILLING ALL PROCESSES ON PORT {port}...", "WARNING")
    
    killed_something = False
    
    try:
        # Method 1: lsof with TCP/UDP variants
        for protocol in ['tcp', 'udp']:
            try:
                pids_cmd = f"lsof -i {protocol}:{port} -t 2>/dev/null || echo ''"
                pids = subprocess.check_output(pids_cmd, shell=True, text=True).strip()
                
                if pids:
                    pids_list = [pid.strip() for pid in pids.split('\n') if pid.strip()]
                    log_message("SCRIPT", f"🎯 Found {protocol.upper()} processes on port {port}: {pids_list}", "WARNING")
                    
                    for pid in pids_list:
                        try:
                            # Kill with extreme prejudice
                            subprocess.run(f"kill -9 {pid} 2>/dev/null", shell=True, timeout=2)
                            killed_something = True
                            log_message("SCRIPT", f"💀 KILLED PID {pid} ({protocol.upper()})", "WARNING")
                        except:
                            pass
            except:
                pass
        
        # Method 2: netstat approach (alternative detection)
        try:
            netstat_cmd = f"netstat -tlnp 2>/dev/null | grep ':{port} ' | awk '{{print $7}}' | cut -d'/' -f1"
            pids = subprocess.check_output(netstat_cmd, shell=True, text=True).strip()
            
            if pids:
                pids_list = [pid.strip() for pid in pids.split('\n') if pid.strip() and pid.strip() != '-']
                log_message("SCRIPT", f"🎯 Found NETSTAT processes on port {port}: {pids_list}", "WARNING")
                
                for pid in pids_list:
                    try:
                        subprocess.run(f"kill -9 {pid} 2>/dev/null", shell=True, timeout=2)
                        killed_something = True
                        log_message("SCRIPT", f"💀 KILLED NETSTAT PID {pid}", "WARNING")
                    except:
                        pass
        except:
            pass
        
        # Method 3: ss command (modern netstat replacement)
        try:
            ss_cmd = f"ss -tlnp 'sport = :{port}' | grep -v State | awk '{{print $6}}' | cut -d',' -f2 | cut -d'=' -f2"
            pids = subprocess.check_output(ss_cmd, shell=True, text=True).strip()
            
            if pids:
                pids_list = [pid.strip() for pid in pids.split('\n') if pid.strip()]
                log_message("SCRIPT", f"🎯 Found SS processes on port {port}: {pids_list}", "WARNING")
                
                for pid in pids_list:
                    try:
                        subprocess.run(f"kill -9 {pid} 2>/dev/null", shell=True, timeout=2)
                        killed_something = True
                        log_message("SCRIPT", f"💀 KILLED SS PID {pid}", "WARNING")
                    except:
                        pass
        except:
            pass
        
        # Method 4: fuser - nuclear option
        try:
            log_message("SCRIPT", f"☢️ NUCLEAR OPTION: Using fuser to kill port {port}", "WARNING")
            subprocess.run(f"fuser -k {port}/tcp 2>/dev/null || true", shell=True, timeout=5)
            subprocess.run(f"fuser -k {port}/udp 2>/dev/null || true", shell=True, timeout=5)
            killed_something = True
        except:
            pass
        
        # Method 5: Docker container detection and killing
        try:
            log_message("SCRIPT", f"🐳 DOCKER NUCLEAR OPTION: Checking for containers on port {port}", "WARNING")
            
            # Find containers using the port
            docker_cmd = f"docker ps --filter 'publish={port}' --format '{{{{.ID}}}}' 2>/dev/null || echo ''"
            container_ids = subprocess.check_output(docker_cmd, shell=True, text=True).strip()
            
            if container_ids:
                container_list = [cid.strip() for cid in container_ids.split('\n') if cid.strip()]
                log_message("SCRIPT", f"🐳 Found Docker containers using port {port}: {container_list}", "WARNING")
                
                for container_id in container_list:
                    try:
                        # Force stop and remove container
                        subprocess.run(f"docker stop {container_id} 2>/dev/null", shell=True, timeout=10)
                        subprocess.run(f"docker rm {container_id} 2>/dev/null", shell=True, timeout=5)
                        killed_something = True
                        log_message("SCRIPT", f"🐳 KILLED DOCKER CONTAINER {container_id}", "WARNING")
                    except:
                        pass
            
            # Also check for port mapping in any format
            broader_docker_cmd = f"docker ps --format 'table {{{{.ID}}}}\\t{{{{.Ports}}}}' | grep ':{port}' | awk '{{print $1}}' || echo ''"
            broader_containers = subprocess.check_output(broader_docker_cmd, shell=True, text=True).strip()
            
            if broader_containers:
                broader_list = [cid.strip() for cid in broader_containers.split('\n') if cid.strip() and cid != "CONTAINER"]
                for container_id in broader_list:
                    try:
                        subprocess.run(f"docker stop {container_id} 2>/dev/null", shell=True, timeout=10)
                        subprocess.run(f"docker rm {container_id} 2>/dev/null", shell=True, timeout=5)
                        killed_something = True
                        log_message("SCRIPT", f"🐳 KILLED BROADER DOCKER CONTAINER {container_id}", "WARNING")
                    except:
                        pass
            
            # Check for docker-compose services in monitoring directory
            monitoring_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitoring")
            if os.path.exists(monitoring_dir):
                try:
                    log_message("SCRIPT", f"🐳 STOPPING DOCKER-COMPOSE IN MONITORING DIR", "WARNING")
                    subprocess.run("docker-compose down --remove-orphans", shell=True, cwd=monitoring_dir, timeout=30)
                    killed_something = True
                except:
                    pass
            
        except Exception as e:
            log_message("SCRIPT", f"Docker killing failed: {e}", "WARNING")
        
        if killed_something:
            log_message("SCRIPT", f"💀 KILLED PROCESSES - Waiting for port cleanup...", "WARNING")
            time.sleep(2)  # Give the OS time to clean up
        
        # Final verification
        if check_port_available(port):
            log_message("SCRIPT", f"✅ PORT {port} SUCCESSFULLY FREED!", "SUCCESS")
            return True
        else:
            log_message("SCRIPT", f"💥 PORT {port} STILL OCCUPIED AFTER NUCLEAR ASSAULT", "ERROR")
            
            # Last resort: try to show what's still using it
            try:
                remaining = subprocess.check_output(f"lsof -i :{port} 2>/dev/null || echo 'Nothing found'", shell=True, text=True)
                log_message("SCRIPT", f"Remaining processes: {remaining.strip()}", "ERROR")
            except:
                pass
            
            return False
    except Exception as e:
        log_message("SCRIPT", f"CRITICAL ERROR while trying to free port {port}: {str(e)}", "ERROR")
        log_debug("SCRIPT", traceback.format_exc())
        return False

def start_frontend(port, build_mode=False):
    """Start the frontend server"""
    log_message("FRONTEND", f"Ensuring port {port} is free...", "INFO")
    
    # Ultra-aggressive port freeing with multiple retries
    if not ensure_port_free_with_retries(port, "Frontend "):
        return None

    client_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client")
    os.chdir(client_dir)
    
    # Always use development mode to ensure proxy settings apply
    # This ensures API requests get properly forwarded to the backend
    log_message("FRONTEND", f"Starting development frontend on port {port}...", "INFO")
    cmd = ["npm", "run", "dev", "--", f"--port={port}"]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=False,
        bufsize=1
    )
    running_processes.append(process)
    
    # Small delay to check if process started correctly
    time.sleep(1)
    if process.poll() is not None:
        log_message("FRONTEND", "Failed to start", "ERROR")
        return None
    
    log_message("FRONTEND", f"Started with PID: {process.pid}", "INFO")
    return process

def start_backend(port):
    """
    Start the backend server with more detailed logging and process monitoring.
    """
    log_message("BACKEND", f"Ensuring port {port} is free...", "INFO")
    
    # Ultra-aggressive port freeing with multiple retries for backend
    if not ensure_port_free_with_retries(port, "Backend "):
        return None
    
    log_message("BACKEND", "Starting uvicorn server with streaming optimizations...", "INFO")
    
    # Prepare the backend environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered
    
    # Add verbose flags to the uvicorn command with extra debugging
    backend_cmd = [
        "/home/jman/miniconda3/bin/python3.12", "-u",  # -u for unbuffered output
        "-m", "uvicorn", 
        "main:app",  # Run main:app from within backend directory
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--log-level", "debug",  # Set uvicorn to debug log level
        "--access-log"  # Enable access logging
    ]
    
    # Fixed: Run from backend directory
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    log_debug("BACKEND", f"Starting command: {' '.join(backend_cmd)} in directory: {backend_dir}")
    
    try:
        # Use text=True to get string output rather than bytes
        process = subprocess.Popen(
            backend_cmd,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately for better diagnostics
            text=True,  # Use text mode instead of binary
            bufsize=1,
            env=env
        )
        
        running_processes.append(process)
        
        # Wait a moment to see if process starts successfully
        time.sleep(0.5)
        
        # Check if process started successfully
        if process.poll() is not None:
            log_message("BACKEND", f"Process exited immediately with code {process.poll()}", "ERROR")
            # Capture any output that may have been generated
            try:
                stdout, stderr = process.communicate(timeout=1)
                if stdout:
                    log_message("BACKEND", f"STDOUT: {stdout.strip()}", "ERROR")
                if stderr:
                    log_message("BACKEND", f"STDERR: {stderr.strip()}", "ERROR")
            except:
                pass
            
            # Also check if we can import the module directly
            log_message("BACKEND", "Testing direct import of main module...", "DEBUG")
            try:
                os.chdir(backend_dir)
                subprocess.run([
                    "python3.12", "-c", 
                    "import sys; print('Python path:', sys.path); import main; print('Import successful')"
                ], check=True, capture_output=True, text=True, timeout=5)
            except subprocess.CalledProcessError as e:
                log_message("BACKEND", f"Direct import test failed: {e.stderr}", "ERROR")
            except Exception as e:
                log_message("BACKEND", f"Direct import test error: {str(e)}", "ERROR")
            
            return None
            
        log_message("BACKEND", f"Started with PID: {process.pid}", "INFO")
        
        # Start a thread to continuously read and log output from the backend
        backend_output_thread = threading.Thread(
            target=stream_process_output,
            args=(process, "BACKEND"),
            daemon=True
        )
        backend_output_thread.start()
        
        return process
    except Exception as e:
        log_message("BACKEND", f"Failed to start: {e}", "ERROR")
        log_debug("BACKEND", traceback.format_exc())
        return None

def wait_for_backend_ready(port, max_attempts=45):
    """
    Wait for the backend API to respond with a 200 status code.
    Enhanced with more comprehensive diagnostics.
    """
    log_message("SCRIPT", "Waiting for backend to be fully operational...", "INFO")
    
    # First, wait for the port to be open
    attempt = 1
    while attempt <= max_attempts:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            s.close()
            
            if result == 0:
                log_message("SCRIPT", f"Port {port} is now open", "INFO")
                break
            else:
                log_message("SCRIPT", f"Port {port} is not open yet (attempt {attempt}/{max_attempts})", "INFO")
                attempt += 1
                time.sleep(2)
        except Exception as e:
            log_message("SCRIPT", f"Error checking port status: {str(e)}", "WARNING")
            attempt += 1
            time.sleep(2)
    
    if attempt > max_attempts:
        log_message("SCRIPT", f"Backend failed to open port {port} after {max_attempts} attempts", "ERROR")
        return False
    
    # Now that the port is open, wait for the API to be ready
    attempt = 1
    while attempt <= max_attempts:
        try:
            conn = http.client.HTTPConnection('localhost', port, timeout=2)
            # Try both endpoints to find one that works
            endpoints = [
                "/api/health",
                "/docs",  # Swagger docs should load if the server is running
                "/"       # Try root as last resort
            ]
            
            success = False
            for endpoint in endpoints:
                try:
                    log_debug("SCRIPT", f"Checking backend API endpoint: {endpoint}")
                    conn.request("GET", endpoint)
                    response = conn.getresponse()
                    status = response.status
                    # Read response data to properly close the connection
                    response.read()
                    
                    if status == 200:
                        log_message("SCRIPT", f"Backend API is ready! Endpoint {endpoint} returned 200 OK", "SUCCESS")
                        success = True
                        break
                    else:
                        log_message("SCRIPT", f"Backend API endpoint {endpoint} returned non-200 status: {status}", "INFO")
                except Exception as endpoint_e:
                    log_debug("SCRIPT", f"Error checking endpoint {endpoint}: {str(endpoint_e)}")
                
            if success:
                return True
                
            log_message("SCRIPT", f"Waiting for backend API... ({attempt}/{max_attempts})", "INFO")
            attempt += 1
            time.sleep(2)
            
        except Exception as e:
            log_message("SCRIPT", f"Error connecting to backend API: {str(e)}", "WARNING")
            log_debug("SCRIPT", traceback.format_exc())
            attempt += 1
            time.sleep(2)
    
    log_message("SCRIPT", f"Backend API failed to respond after {max_attempts} attempts", "ERROR")
    return False

def main():
    """Main function to run both servers"""
    global verbose_mode, log_file
    
    parser = argparse.ArgumentParser(description="Run frontend and backend servers with synchronized output")
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT, help="Port for the frontend server")
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT, help="Port for the backend server")
    parser.add_argument("--build", action="store_true", help="Run frontend in production build mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debugging output")
    parser.add_argument("--no-log-file", action="store_true", help="Disable logging to file")
    
    args = parser.parse_args()
    verbose_mode = args.verbose
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
            print(f"Created logs directory at {LOG_DIR}")
        except Exception as e:
            print(f"Error creating logs directory: {e}")
            return 1
    
    # Setup log file (overwrite mode)
    if not args.no_log_file:
        try:
            log_file = open(LOG_FILE, 'w', encoding='utf-8')
            print(f"Logging to file: {LOG_FILE}")
        except Exception as e:
            print(f"Error opening log file: {e}")
            return 1
    
    # Setup comprehensive signal handlers for clean exit
    log_message("SCRIPT", "Setting up signal handlers for graceful shutdown...", "INFO")
    signal.signal(signal.SIGINT, cleanup)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # Termination signal
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, cleanup)   # Hang up signal (Unix only)
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, cleanup)  # Quit signal (Unix only)
    
    try:
        # Kill any existing processes on our ports before starting (only if occupied)
        log_message("SCRIPT", "Checking ports before startup...", "INFO")
        ensure_port_free_with_retries(args.frontend_port, "Frontend ")
        ensure_port_free_with_retries(args.backend_port, "Backend ")
        time.sleep(1)  # Give system time to release ports if any were freed
        
        # Source .bashrc if it exists (for environment variables)
        log_message("SCRIPT", "Setting up environment...", "INFO")
        bashrc_path = os.path.expanduser("~/.bashrc")
        if os.path.exists(bashrc_path):
            try:
                subprocess.run(f"source {bashrc_path}", shell=True, executable="/bin/bash")
            except Exception as e:
                log_message("SCRIPT", f"Warning: Failed to source .bashrc: {e}", "WARNING")
        
        # Start frontend first
        frontend_process = start_frontend(args.frontend_port, args.build)
        if not frontend_process:
            cleanup()
            return 1
        
        # Start backend with extra logging
        log_message("SCRIPT", "Starting backend server with detailed monitoring...", "INFO")
        backend_process = start_backend(args.backend_port)
        if not backend_process:
            cleanup()
            return 1
        
        # Check if backend API comes online with improved diagnostics
        log_message("SCRIPT", "Waiting for backend to be fully operational...", "INFO")
        log_message("SCRIPT", "NOTE: Backend loads large AI models which can take 2-5 minutes on first start", "WARNING")
        if not wait_for_backend_ready(args.backend_port, max_attempts=180):  # Increased to 6 minutes
            log_message("SCRIPT", "Backend API failed to start properly, shutting down", "ERROR")
            cleanup()
            return 1
        
        log_message("SCRIPT", "All servers started successfully!", "SUCCESS")
        log_message("SCRIPT", f"Frontend: http://localhost:{args.frontend_port}", "INFO")
        log_message("SCRIPT", f"Backend: http://localhost:{args.backend_port}", "INFO")
        log_message("SCRIPT", "Press Ctrl+C to stop all servers", "INFO")
        
        # Create threads to stream output
        frontend_thread = threading.Thread(
            target=stream_output,
            args=(frontend_process, "FRONTEND", "FRONTEND"),
            daemon=True
        )
        
        frontend_thread.start()
        
        # Wait for processes to exit
        while True:
            if frontend_process.poll() is not None:
                log_message("SCRIPT", "Frontend server exited, shutting down", "INFO")
                break
                
            if backend_process.poll() is not None:
                log_message("SCRIPT", "Backend server exited, shutting down", "INFO")
                break
                
            time.sleep(1)
        
        cleanup()
        
    except KeyboardInterrupt:
        log_message("SCRIPT", "🛑 Keyboard interrupt received (Ctrl+C)", "WARNING")
        cleanup()
        return 0
    except Exception as e:
        log_message("SCRIPT", f"💥 Unexpected error: {e}", "ERROR")
        log_debug("SCRIPT", traceback.format_exc())
        cleanup()
        return 1
    finally:
        # Only run cleanup if it hasn't been performed yet
        if not shutdown_performed:
            log_message("SCRIPT", "🔄 Running cleanup in finally block...", "INFO")
            cleanup()

if __name__ == "__main__":
    sys.exit(main()) 
