import os
import datetime
import argparse
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

def extract_arguments_from_python_file(file_path):
    arguments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex to find add_argument calls
        # This is a simplified regex and might not catch all cases
        arg_pattern = re.compile(r'parser\\.add_argument\\((.*?)\\)', re.DOTALL)
        
        for match in arg_pattern.finditer(content):
            arg_str = match.group(1)
            arg_name_match = re.search(r"^\\s*['\"](--[a-zA-Z0-9\\-_]+)['\"]", arg_str)
            help_match = re.search(r"help\\s*=\\s*['\"](.*?)['\"]", arg_str)
            
            name = arg_name_match.group(1) if arg_name_match else "Unknown"
            help_text = help_match.group(1) if help_match else "No description."
            
            arguments.append(f"`{name}`: {help_text}")
    except Exception as e:
        arguments.append(f"Error extracting arguments: {e}")
    return arguments

# Argument parser
parser = argparse.ArgumentParser(description="Update dev-test-log.MD")
parser.add_argument('--model', type=str, default='N/A', help='Name of the model being tested')
parser.add_argument('--segment-size', type=str, default='N/A', help='Segment size used in test')
parser.add_argument('--limit', type=int, default=None, help='Optional number of most recent files to log')
parser.add_argument('--days', type=int, default=1, help='Number of days to look back for file modifications (default: 1 for today)')
parser.add_argument('--debug', action='store_true', help='Show verbose processing output')
args = parser.parse_args()

# Constants
directory_path = "./tests/dev"
log_file_path = os.path.join(directory_path, "dev-test-log.MD")

if not os.path.exists(directory_path):
    raise FileNotFoundError(f"Directory not found: {directory_path}")

# Collect file info
file_infos = []
for fname in os.listdir(directory_path):
    full_path = os.path.join(directory_path, fname)
    if os.path.isfile(full_path) and fname != "dev-test-log.MD":
        mtime = os.path.getmtime(full_path)
        file_infos.append((fname, mtime, full_path))

# Filter by date (today only by default)
cutoff_date = datetime.datetime.now() - datetime.timedelta(days=args.days)
file_infos = [(f, m, p) for f, m, p in file_infos if datetime.datetime.fromtimestamp(m) > cutoff_date]

# Sort by modification time (newest first)
file_infos.sort(key=lambda x: x[1], reverse=True)

if args.limit:
    file_infos = file_infos[:args.limit]

# Group files by date and format log entries
entries_by_date = defaultdict(list)
for fname, mtime, full_path in file_infos:
    dt = datetime.datetime.fromtimestamp(mtime)
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S")
    
    description = get_file_description(full_path)
    arguments = []
    if full_path.endswith('.py'):
        arguments = extract_arguments_from_python_file(full_path)
    
    # Construct the description line including model and segment size
    desc_line = f"Model: `{args.model}`, Segment Size: `{args.segment_size}`"
    
    entries_by_date[date_str].append({
        "time": time_str,
        "fname": fname,
        "description": description,
        "desc_line": desc_line,
        "arguments": arguments,
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
        if entry['arguments']:
            lines.append(f"Arguments:")
            for arg in entry['arguments']:
                lines.append(f"  - {arg}")
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

if args.debug:
    print(f"✅ Updated {log_file_path} with {len(file_infos)} files.")