#!/bin/zsh

set -u

project_dir="${0:A:h}"
streamlit_file="$project_dir/streamlit_app.py"
pid_file="/tmp/operations-knowledge-copilot-streamlit.pid"
ollama_pid_file="/tmp/operations-knowledge-copilot-ollama.pid"
stopped_streamlit=false

if [[ -f "$pid_file" ]]; then
    saved_pid="$(<"$pid_file")"
    if [[ "$saved_pid" == <-> ]]; then
        saved_command="$(/bin/ps -p "$saved_pid" -o command= 2>/dev/null || true)"
        if [[ "$saved_command" == *"$project_dir/.venv/bin/streamlit"* && "$saved_command" == *"streamlit_app.py"* ]]; then
            /bin/kill "$saved_pid" 2>/dev/null || true
            stopped_streamlit=true
        fi
    fi
    /bin/rm -f "$pid_file"
fi

if [[ -f "$ollama_pid_file" ]]; then
    ollama_pid="$(<"$ollama_pid_file")"
    if [[ "$ollama_pid" == <-> ]]; then
        ollama_command="$(/bin/ps -p "$ollama_pid" -o command= 2>/dev/null || true)"
        if [[ "$ollama_command" == *"ollama"*"serve"* ]]; then
            /bin/kill "$ollama_pid" 2>/dev/null || true
        fi
    fi
    /bin/rm -f "$ollama_pid_file"
fi

if [[ "$stopped_streamlit" == true ]]; then
    /usr/bin/osascript -e 'display notification "The browser interface has stopped." with title "Operations Knowledge Copilot"'
else
    /usr/bin/osascript -e 'display alert "Operations Knowledge Copilot" message "No Streamlit process started by START_APP.command was found. If you started it manually, stop it in Terminal with Control-C."'
fi
