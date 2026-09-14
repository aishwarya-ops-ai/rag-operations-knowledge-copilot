#!/bin/zsh

set -u

project_dir="${0:A:h}"
streamlit_command="$project_dir/.venv/bin/streamlit"
streamlit_file="$project_dir/streamlit_app.py"
pid_file="/tmp/operations-knowledge-copilot-streamlit.pid"
log_file="/tmp/operations-knowledge-copilot-streamlit.log"
ollama_pid_file="/tmp/operations-knowledge-copilot-ollama.pid"
ollama_log_file="/tmp/operations-knowledge-copilot-ollama.log"
app_url="http://localhost:8501"
health_url="$app_url/_stcore/health"

if [[ ! -x "$streamlit_command" ]]; then
    /usr/bin/osascript -e 'display alert "Operations Knowledge Copilot could not start" message "The .venv environment or Streamlit is missing. Reinstall the project dependencies first." as critical'
    exit 1
fi

if [[ ! -f "$streamlit_file" ]]; then
    /usr/bin/osascript -e 'display alert "Operations Knowledge Copilot could not start" message "streamlit_app.py was not found in the project folder." as critical'
    exit 1
fi

if ! /usr/bin/curl --fail --silent "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    if ! /usr/bin/open -gj -a Ollama >/dev/null 2>&1; then
        ollama_command="$(command -v ollama 2>/dev/null || true)"
        if [[ -n "$ollama_command" ]]; then
            /usr/bin/nohup "$ollama_command" serve >"$ollama_log_file" 2>&1 &
            printf '%s\n' "$!" >"$ollama_pid_file"
        fi
    fi

    for attempt in {1..15}; do
        if /usr/bin/curl --fail --silent "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
            break
        fi
        /bin/sleep 1
    done
fi

if ! /usr/bin/curl --fail --silent "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    /usr/bin/osascript -e 'display alert "Ollama is not responding" message "The browser app will open, but grounded answer generation will use the safe fallback until Ollama is running." as warning'
fi

if [[ -f "$pid_file" ]]; then
    saved_pid="$(<"$pid_file")"
    if [[ "$saved_pid" == <-> ]]; then
        saved_command="$(/bin/ps -p "$saved_pid" -o command= 2>/dev/null || true)"
        if [[ "$saved_command" == *"$streamlit_command"* && "$saved_command" == *"streamlit_app.py"* ]]; then
            /usr/bin/open "$app_url"
            exit 0
        fi
    fi
    /bin/rm -f "$pid_file"
fi

if /usr/bin/curl --fail --silent "$health_url" >/dev/null 2>&1; then
    existing_pid="$(/usr/sbin/lsof -tiTCP:8501 -sTCP:LISTEN 2>/dev/null | /usr/bin/head -n 1)"
    existing_command="$(/bin/ps -p "$existing_pid" -o command= 2>/dev/null || true)"
    if [[ "$existing_pid" == <-> && "$existing_command" == *"$streamlit_command"* && "$existing_command" == *"streamlit_app.py"* ]]; then
        printf '%s\n' "$existing_pid" >"$pid_file"
        /usr/bin/open "$app_url"
        exit 0
    fi
    /usr/bin/osascript -e 'display alert "Port 8501 is already in use" message "Another application is using the browser app port, so Operations Knowledge Copilot was not started." as critical'
    exit 1
fi

cd "$project_dir" || exit 1
/usr/bin/nohup "$streamlit_command" run "$streamlit_file" \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    >"$log_file" 2>&1 &
streamlit_pid=$!
printf '%s\n' "$streamlit_pid" >"$pid_file"

for attempt in {1..30}; do
    if /usr/bin/curl --fail --silent "$health_url" >/dev/null 2>&1; then
        /usr/bin/open "$app_url"
        /usr/bin/osascript -e 'display notification "The browser interface is ready." with title "Operations Knowledge Copilot"'
        exit 0
    fi
    /bin/sleep 1
done

if /bin/kill -0 "$streamlit_pid" 2>/dev/null; then
    /bin/kill "$streamlit_pid" 2>/dev/null || true
fi
/bin/rm -f "$pid_file"
/usr/bin/osascript -e 'display alert "Operations Knowledge Copilot could not start" message "Streamlit did not become ready. See /tmp/operations-knowledge-copilot-streamlit.log for details." as critical'
exit 1
