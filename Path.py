from pathlib import Path
file = Path(__file__)
print(type(file))
print(file)
print(file.parent)
print()
for p in file.parents:
    print(p)
    
print(file.parent / 'hell.py')