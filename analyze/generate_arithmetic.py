import re
import numpy as np
import tqdm

def evaluate_expression(expression):
    # Step 1: Add spaces around operators for clarity
    expression = re.sub(r'([+\-*/])', r' \1 ', expression)
    
    # Step 2: Split the expression into parts
    parts = expression.split()
    
    # Step 3: Initialize the steps list
    steps = []
    steps.append(expression)
    
    # Step 4: Handle multiplication and division first
    while '*' in parts or '/' in parts:
        for i in range(len(parts)):
            if parts[i] == '*':
                result = int(parts[i-1]) * int(parts[i+1])
                parts[i-1:i+2] = [str(result)]
                step_expression = ' '.join(parts)
                steps.append(step_expression)
                break
            elif parts[i] == '/':
                result = int(parts[i-1]) / int(parts[i+1])
                parts[i-1:i+2] = [str(result)]
                step_expression = ' '.join(parts)
                steps.append(step_expression)
                break
    
    # Step 5: Handle addition and subtraction
    while '+' in parts or '-' in parts:
        for i in range(len(parts)):
            if parts[i] == '+':
                result = int(parts[i-1]) + int(parts[i+1])
                parts[i-1:i+2] = [str(result)]
                step_expression = ' '.join(parts)
                steps.append(step_expression)
                break
            elif parts[i] == '-':
                result = int(parts[i-1]) - int(parts[i+1])
                parts[i-1:i+2] = [str(result)]
                step_expression = ' '.join(parts)
                steps.append(step_expression)
                break
    
    return steps



def generate_input(num_steps):
    input_str = str(np.random.randint(0, high=100))
    
    for i in range(num_steps):
        operator = np.random.choice(['+', '-', '*'])
        operand = str(np.random.randint(0, high=100))
        input_str += ' ' + operator + ' ' + operand
    return input_str



dataset =  []
for j in tqdm.tqdm(range(10000)):
    