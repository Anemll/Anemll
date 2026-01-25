import os
import datetime
import re
from collections import defaultdict

def get_file_description(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to extract Python docstring
        if file_path.endswith('.py'):
            match = re.search(r'"""\n(.*?)\n"""', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # For Markdown files, take the first few lines
        if file_path.endswith('.md'):
            lines = content.split('\n')
            description_lines = []
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    description_lines.append(line.strip())
                if len(description_lines) >= 3 or (line.strip() and line.strip().startswith('---')):
                    break
            return ' '.join(description_lines).strip()
        
        # For JSON files, read the first few lines
        if file_path.endswith('.json'):
            lines = content.split('\n')
            return 'JSON data: ' + ' '.join(lines[:3]).strip()
            
        # Default: return first line or generic description
        first_line = content.split('\n')[0].strip()
        if first_line:
            return f"File content starts with: '{first_line}'"
        return "No specific description found."
    except Exception as e:
        return f"Error reading file: {e}"

directory_path = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev"
log_file_path = os.path.join(directory_path, "dev-test-log.MD")

# Collect file info
file_infos = []
for fname in os.listdir(directory_path):
    full_path = os.path.join(directory_path, fname)
    if os.path.isfile(full_path) and fname != "dev-test-log.MD":
        mtime = os.path.getmtime(full_path)
        file_infos.append((fname, mtime, full_path))

# Sort by modification time (newest first)
file_infos.sort(key=lambda x: x[1], reverse=True)

# Group files by date and format log entries
entries_by_date = defaultdict(list)
for fname, mtime, full_path in file_infos:
    dt = datetime.datetime.fromtimestamp(mtime)
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S")
    
    description = get_file_description(full_path)
    
    # For now, we'll use dummy model and segment size as they are not extracted from files
    # The user's previous request implies these are external parameters.
    # If the user wants to extract these from file content, that would require more specific logic.
    desc_line = f"Model: `N/A`, Segment Size: `N/A`"
    
    entries_by_date[date_str].append({
        "time": time_str,
        "fname": fname,
        "description": description,
        "desc_line": desc_line,
        "full_path": full_path # Keep full path for grouping
    })

# Format log
lines = ["# Development Test Log", "", "This log tracks daily testing activities. New entries should be added to the top.", ""]

for date, entries in entries_by_date.items():
    lines.append(f"---")
    lines.append(f"## {date}")
    lines.append("")
    
    for entry in entries:
        lines.append(f"**{entry['time']}** `{entry['fname']}`")
        lines.append(f"{entry['desc_line']}")
        lines.append(f"Description: {entry['description']}")
        lines.append("")
    
    lines.append("files updated:")
    
    # Group by subdirectory if applicable
    path_groups = defaultdict(list)
    for entry in entries:
        rel_path = os.path.relpath(entry['full_path'], start=directory_path)
        parent = os.path.dirname(rel_path)
        path_groups[parent].append(entry['fname'])

    for parent, fnames in path_groups.items():
        lines.append(f"- `{os.path.join('tests/dev', parent)}/`: " + ", ".join(f"`{f}`" for f in fnames))
    lines.append("")

# Overwrite the log file
with open(log_file_path, "w", encoding='utf-8') as f:
    f.write("\n".join(lines))

print(f"Successfully updated {log_file_path} with detailed information.")