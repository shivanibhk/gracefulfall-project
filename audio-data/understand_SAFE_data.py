from pathlib import Path 
from collections import Counter 
files=Path('audio-data/SAFE_raw').glob('*.wav') 
print(Counter(p.stem.split('-')[-1] for p in files)) #Counter({'02': 475, '01': 475})

## SAFE data wav files labeled with 01 for fall, 02 for non fall.