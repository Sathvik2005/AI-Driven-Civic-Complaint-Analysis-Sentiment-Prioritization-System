#!/usr/bin/env python3
"""Remove emojis from Jupyter notebook and make it professional"""

import json
import re

# Read the notebook
with open('Week1_Data_Collection_Cleaning_EDA.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define emoji patterns to remove
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+", flags=re.UNICODE)

# Professional replacements
replacements = {
    '📋': '[INFO]',
    '📥': '[DOWNLOAD]',
    '✅': '[SUCCESS]',
    '📂': '[FILES]',
    '⚠️': '[WARNING]',
    '⚠': '[WARNING]',
    'ℹ️': '[INFO]',
    'ℹ': '[INFO]',
    '🔄': '[PROCESSING]',
    '💡': '[TIP]',
    '📊': '[STATS]',
    '📏': '[SIZE]',
    '💾': '[SAVE]',
    '📅': '[DATE]',
    '🔍': '[PREVIEW]',
    '📑': '[LIST]',
    '🎯': '[TARGET]',
    '📝': '[TEXT]',
    '🏢': '[DEPT]',
    '📈': '[CHART]',
    '🧹': '[CLEAN]',
    '🧪': '[TEST]',
    '🎨': '[VISUAL]',
    '🔤': '[WORDS]',
    '💬': '[PHRASE]',
    '🚀': '[COMPLETE]',
    '🏆': '[TOP]',
    '📄': '[FILE]',
    '🔧': '[GIT]',
    '👤': '[USER]',
    '✓': '[OK]',
    '🔢': '[TYPE]',
}

# Process each cell
for cell in notebook['cells']:
    if cell['cell_type'] in ['code', 'markdown']:
        # Process source
        if isinstance(cell['source'], list):
            new_source = []
            for line in cell['source']:
                # Apply replacements
                for emoji, replacement in replacements.items():
                    line = line.replace(emoji, replacement)
                # Remove any remaining emojis
                line = emoji_pattern.sub('', line)
                new_source.append(line)
            cell['source'] = new_source
        else:
            # Apply replacements
            for emoji, replacement in replacements.items():
                cell['source'] = cell['source'].replace(emoji, replacement)
            # Remove any remaining emojis
            cell['source'] = emoji_pattern.sub('', cell['source'])

# Write back the notebook
with open('Week1_Data_Collection_Cleaning_EDA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("[SUCCESS] Emojis removed and notebook professionalized!")
print("[INFO] Total cells processed:", len(notebook['cells']))
