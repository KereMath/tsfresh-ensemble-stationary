import codecs
import os

files = ['confusion_matrix.txt', 'feature_importances.txt']
for name in files:
    if not os.path.exists(name):
        print(f"File not found: {name}")
        continue
        
    try:
        # Try converting from utf-16
        with codecs.open(name, 'r', encoding='utf-16') as f:
            content = f.read()
            
        with codecs.open(name, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Converted {name} to UTF-8")
    except Exception as e:
        print(f"Error converting {name}: {e}")
