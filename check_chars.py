#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open('ppocr_keys_v1.txt', 'r', encoding='utf-8') as f:
    chars = f.read().strip().split('\n')

print(f'Total characters: {len(chars)}')
print(f'First 20 characters: {chars[:20]}')

# Check specific indices from our OCR output
test_indices = [245, 1217, 1958, 3332, 1033, 4902, 3538, 4547, 4548, 3588, 4245, 5233, 466, 3537]
print(f'\nChecking specific indices from OCR output:')
for idx in test_indices:
    if idx < len(chars):
        print(f'Index {idx}: "{chars[idx]}"')
    else:
        print(f'Index {idx}: OUT OF RANGE (max {len(chars)-1})')

# Show range around numbers/letters
print(f'\nSample around ASCII range:')
for i in range(max(0, 48-5), min(len(chars), 58+5)):  # Around numbers 0-9
    print(f'Index {i}: "{chars[i]}"')