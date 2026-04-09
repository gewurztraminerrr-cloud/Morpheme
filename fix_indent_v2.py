import sys

file_path = 'board_generator.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.lstrip()
    if not stripped:
        new_lines.append('\n')
        continue
    
    leading_spaces = len(line) - len(stripped)
    
    # If indentation is greater than 20, it's likely corrupted.
    # We'll cap the common corruptions.
    if leading_spaces > 30:
         # Reduce by a factor of 4 or something?
         # Or just use the original intent.
         # This is too hard.
         pass

    # Better: just fix the specific methods I know are broken.
    if 'def get_difficulty_label' in stripped:
         new_lines.append('    ' + stripped)
         continue
    if '"""Derive difficulty label' in stripped:
         new_lines.append('        ' + stripped)
         continue
    if 'is_large = (rows * cols >= 35)' in stripped:
         new_lines.append('        ' + stripped)
         continue
    
    new_lines.append(line)

# Wait! The easiest way is to use `autopep8` or `black` if available.
# Let's check if black is installed.
import subprocess
try:
    subprocess.run(['black', '--version'], capture_output=True)
    has_black = True
except:
    has_black = False

if has_black:
    subprocess.run(['black', '-l', '120', file_path])
    print("Formatted with black")
else:
    # Manual fix for the most obvious corruption
    with open(file_path, 'w') as f:
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                f.write('\n')
                continue
            
            # Simple rule: if it was indented 16+, it's probably 8+
            leading = len(line) - len(stripped)
            if leading >= 16:
                 # It was probably doubled or tripled
                 # We'll try to find the class/method context.
                 # Let's just fix the prefix specifically.
                 if line.startswith(' ' * 16):
                      f.write('    ' * 2 + stripped) # 8 spaces
                 elif line.startswith(' ' * 24):
                      f.write('    ' * 3 + stripped) # 12 spaces
                 else:
                      f.write(line)
            else:
                 f.write(line)
