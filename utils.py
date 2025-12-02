import re
import json

def structure_logs(raw_text):
    structured = []

    # Updated pattern to match your actual log format
    # Supports: 2025-12-01T19:32:04.313634 [2504] ERROR: ...
    text_log_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"  # Timestamp with optional .microseconds
        r"(?:\s+\[\d+\])?"  # Optional [process_id]
        r"\s+\[?(\w+)\]?:?\s+(.*)"  # Level and message
    )

    current_log = None
    line_count = 0

    # Process line by line without creating huge list in memory
    for line in raw_text.split('\n'):  # Faster than splitlines()
        line_count += 1
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Try JSON log (fast path - skip if not JSON)
        if line.startswith('{'):
            try:
                parsed = json.loads(line)
                if all(k in parsed for k in ("timestamp", "level", "message")):
                    if current_log:
                        structured.append(current_log)
                    current_log = {
                        "timestamp": parsed["timestamp"],
                        "level": parsed["level"],
                        "message": parsed["message"]
                    }
                    continue
            except json.JSONDecodeError:
                pass

        # Try text log
        match = text_log_pattern.match(line)
        if match:
            if current_log:
                structured.append(current_log)
            current_log = {
                "timestamp": match.group(1),
                "level": match.group(2).upper(),  # Normalize to uppercase
                "message": match.group(3)
            }
        else:
            # Unstructured line: append to current log or create UNKNOWN
            if current_log:
                current_log["message"] += "\n" + line
            else:
                current_log = {
                    "timestamp": "",
                    "level": "UNKNOWN",
                    "message": line
                }

    if current_log:
        structured.append(current_log)

    return structured


def chunk_structured_logs(logs, chunk_size=100, overlap=10):
    """
    Create overlapping chunks to preserve context at boundaries.
    Optimized for large files: 100 logs per chunk instead of 10.
    
    Args:
        logs: List of structured log dicts
        chunk_size: Number of logs per chunk (default 100)
        overlap: Number of logs to overlap between chunks (default 10)
    
    Returns:
        List of formatted chunk strings
    """
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(logs), step):
        chunk = logs[i:i + chunk_size]
        if not chunk:
            break
            
        chunk_text = "\n".join(
            [f"{log['timestamp']} [{log['level']}] {log['message']}" for log in chunk]
        )
        chunks.append(chunk_text)
        
        # Stop if we've covered all logs
        if i + chunk_size >= len(logs):
            break
    
    return chunks
