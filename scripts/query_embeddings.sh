#!/bin/bash

# Neural Consciousness Memory Query Script
# ========================================
# Purpose: Query and analyze memory data stored in Redis Stack
# 
# This script connects to the Redis Stack instance and retrieves 
# memory embeddings, content, and metadata for analysis.

set -e

# Configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}"
}

check_redis_connection() {
    if ! command -v redis-cli &> /dev/null; then
        echo -e "${RED}Error: redis-cli not found. Please install Redis tools.${NC}"
        exit 1
    fi
    
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
        echo -e "${RED}Error: Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT${NC}"
        echo "Make sure Redis Stack is running: cd docker && docker-compose up -d"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Connected to Redis at $REDIS_HOST:$REDIS_PORT${NC}"
}

# Query functions
show_memory_summary() {
    print_section "Memory Storage Summary"
    
    echo "Short-term Memory (STM) entries:"
    stm_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "stm:*" | wc -l)
    echo -e "  ${GREEN}$stm_count${NC} STM records"
    
    echo "Long-term Memory (LTM) entries:"
    ltm_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "ltm:*" | wc -l)
    echo -e "  ${GREEN}$ltm_count${NC} LTM records"
    
    echo "Vector Indexes:"
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" FT.INFO idx:stm_vectors &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} STM vector index (idx:stm_vectors)"
    else
        echo -e "  ${RED}✗${NC} STM vector index missing"
    fi
    
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" FT.INFO idx:ltm_vectors &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} LTM vector index (idx:ltm_vectors)"
    else
        echo -e "  ${RED}✗${NC} LTM vector index missing"
    fi
}

show_recent_memories() {
    local limit=${1:-5}
    print_section "Recent Memory Entries (Last $limit)"
    
    echo "Short-term memories:"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "stm:*" | head -"$limit" | while read -r key; do
        if [ -n "$key" ]; then
            content=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.content 2>/dev/null || echo "null")
            timestamp=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.timestamp 2>/dev/null || echo "null")
            tags=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.tags 2>/dev/null || echo "null")
            
            echo -e "  ${PURPLE}$key${NC}"
            echo -e "    Content: ${YELLOW}$(echo "$content" | tr -d '[]"' | cut -c1-100)...${NC}"
            echo -e "    Timestamp: $timestamp"
            echo -e "    Tags: $tags"
            echo
        fi
    done
    
    echo "Long-term memories:"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "ltm:*" | head -"$limit" | while read -r key; do
        if [ -n "$key" ]; then
            content=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.content 2>/dev/null || echo "null")
            retrieval_score=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.retrieval_score 2>/dev/null || echo "null")
            circle=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.circle 2>/dev/null || echo "null")
            
            echo -e "  ${PURPLE}$key${NC}"
            echo -e "    Content: ${YELLOW}$(echo "$content" | tr -d '[]"' | cut -c1-100)...${NC}"
            echo -e "    Circle: $circle"
            echo -e "    Retrieval Score: $retrieval_score"
            echo
        fi
    done
}

show_memory_circles() {
    print_section "Memory Circles (Categories)"
    
    # Get unique circles from both STM and LTM
    echo "Analyzing memory categories..."
    
    temp_file=$(mktemp)
    
    # Collect circles from STM
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "stm:*" | while read -r key; do
        if [ -n "$key" ]; then
            circle=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.circle 2>/dev/null | tr -d '[]"')
            if [ "$circle" != "null" ] && [ -n "$circle" ]; then
                echo "$circle" >> "$temp_file"
            fi
        fi
    done
    
    # Collect circles from LTM
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "ltm:*" | while read -r key; do
        if [ -n "$key" ]; then
            circle=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.circle 2>/dev/null | tr -d '[]"')
            if [ "$circle" != "null" ] && [ -n "$circle" ]; then
                echo "$circle" >> "$temp_file"
            fi
        fi
    done
    
    # Show unique circles with counts
    if [ -f "$temp_file" ]; then
        sort "$temp_file" | uniq -c | sort -nr | while read -r count circle; do
            echo -e "  ${GREEN}$circle${NC}: $count memories"
        done
        rm "$temp_file"
    fi
}

analyze_embedding_quality() {
    print_section "Embedding Quality Analysis"
    
    echo "Checking embedding dimensions and completeness..."
    
    # Check a sample of embeddings
    sample_keys=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "*:*" | head -5)
    
    for key in $sample_keys; do
        if [ -n "$key" ]; then
            embedding=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.embedding 2>/dev/null)
            if [ "$embedding" != "null" ] && [ -n "$embedding" ]; then
                # Count array elements (rough dimension check)
                dim=$(echo "$embedding" | grep -o "," | wc -l)
                dim=$((dim + 1))
                echo -e "  ${PURPLE}$key${NC}: ${GREEN}$dim${NC} dimensions"
            else
                echo -e "  ${PURPLE}$key${NC}: ${RED}No embedding${NC}"
            fi
        fi
    done
}

search_memories() {
    local search_term="$1"
    print_section "Searching Memories for: '$search_term'"
    
    if [ -z "$search_term" ]; then
        echo "Usage: $0 search <term>"
        return
    fi
    
    echo "Searching in STM and LTM content..."
    
    # Search STM
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "stm:*" | while read -r key; do
        if [ -n "$key" ]; then
            content=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.content 2>/dev/null | tr -d '[]"')
            if echo "$content" | grep -qi "$search_term"; then
                timestamp=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.timestamp 2>/dev/null)
                echo -e "${GREEN}STM Match:${NC} $key"
                echo -e "  Content: ${YELLOW}$(echo "$content" | cut -c1-200)...${NC}"
                echo -e "  Timestamp: $timestamp"
                echo
            fi
        fi
    done
    
    # Search LTM
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "ltm:*" | while read -r key; do
        if [ -n "$key" ]; then
            content=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.content 2>/dev/null | tr -d '[]"')
            if echo "$content" | grep -qi "$search_term"; then
                retrieval_score=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" $.retrieval_score 2>/dev/null)
                echo -e "${GREEN}LTM Match:${NC} $key"
                echo -e "  Content: ${YELLOW}$(echo "$content" | cut -c1-200)...${NC}"
                echo -e "  Retrieval Score: $retrieval_score"
                echo
            fi
        fi
    done
}

export_memory_data() {
    local output_file="${1:-memory_export_$(date +%Y%m%d_%H%M%S).json}"
    print_section "Exporting Memory Data to: $output_file"
    
    echo "Exporting all memory entries..."
    
    {
        echo "{"
        echo "  \"export_timestamp\": \"$(date -Iseconds)\","
        echo "  \"stm_memories\": ["
        
        # Export STM
        first_stm=true
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "stm:*" | while read -r key; do
            if [ -n "$key" ]; then
                if [ "$first_stm" = false ]; then
                    echo ","
                fi
                echo -n "    {"
                echo -n "\"key\": \"$key\", "
                memory_data=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" 2>/dev/null)
                echo -n "\"data\": $memory_data"
                echo -n "}"
                first_stm=false
            fi
        done
        
        echo ""
        echo "  ],"
        echo "  \"ltm_memories\": ["
        
        # Export LTM
        first_ltm=true
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" --scan --pattern "ltm:*" | while read -r key; do
            if [ -n "$key" ]; then
                if [ "$first_ltm" = false ]; then
                    echo ","
                fi
                echo -n "    {"
                echo -n "\"key\": \"$key\", "
                memory_data=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" JSON.GET "$key" 2>/dev/null)
                echo -n "\"data\": $memory_data"
                echo -n "}"
                first_ltm=false
            fi
        done
        
        echo ""
        echo "  ]"
        echo "}"
    } > "$output_file"
    
    echo -e "${GREEN}✓ Export completed: $output_file${NC}"
    echo "File size: $(ls -lh "$output_file" | awk '{print $5}')"
}

show_help() {
    print_header "Neural Consciousness Memory Query Tool"
    echo
    echo "Usage: $0 [command] [arguments]"
    echo
    echo "Commands:"
    echo -e "  ${GREEN}summary${NC}                    Show memory storage summary"
    echo -e "  ${GREEN}recent [limit]${NC}             Show recent memory entries (default: 5)"
    echo -e "  ${GREEN}circles${NC}                    Show memory circles/categories"
    echo -e "  ${GREEN}quality${NC}                    Analyze embedding quality"
    echo -e "  ${GREEN}search <term>${NC}              Search memories for specific term"
    echo -e "  ${GREEN}export [filename]${NC}          Export all memory data to JSON"
    echo -e "  ${GREEN}help${NC}                       Show this help message"
    echo
    echo "Examples:"
    echo "  $0 summary"
    echo "  $0 recent 10"
    echo "  $0 search \"machine learning\""
    echo "  $0 export my_memories.json"
    echo
    echo "Environment Variables:"
    echo "  REDIS_HOST  - Redis host (default: localhost)"
    echo "  REDIS_PORT  - Redis port (default: 6379)"
    echo "  REDIS_DB    - Redis database (default: 0)"
}

# Main execution
main() {
    case "${1:-help}" in
        "summary")
            check_redis_connection
            show_memory_summary
            ;;
        "recent")
            check_redis_connection
            show_recent_memories "${2:-5}"
            ;;
        "circles")
            check_redis_connection
            show_memory_circles
            ;;
        "quality")
            check_redis_connection
            analyze_embedding_quality
            ;;
        "search")
            check_redis_connection
            search_memories "$2"
            ;;
        "export")
            check_redis_connection
            export_memory_data "$2"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"