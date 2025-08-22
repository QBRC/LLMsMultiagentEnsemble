import re
import json


def fix_json(json_string:str, verbose:bool=False):
    """Fix a given flatten json string possibly with grammar errors."""

    json_str = json_string
    
    # if verbose:
    #     print(f'\nOriginal input JSON string:\n{json_str}')
    
    # Regex matches content between ```json and ```
    pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    match = pattern.search(json_str)
    if match:
        json_str = match.group(1)
        
    # # check if the json_str missed both the opening bracket and closing bracket
    if json_str.count('{') == 0 and json_str.count('}') == 0:
        # # replace the first ``` as { and the last ``` as }
        # first_replacement = json_str.replace("```", "{", 1)
        # last_backtick_index = first_replacement.rfind("```")
        # fixed_json_str = first_replacement[:last_backtick_index] + "}" + first_replacement[last_backtick_index+3:]

        ##  Add "{" before the first JSON key-value pair and "}" at the end of the last key-value pair
        # Regular expression to match JSON key-value pairs; (without considering numeric values)
        pattern = r'"[^"]+"\s*:\s*".+?"'
        
        # Find all matches of the pattern
        matches = re.findall(pattern, json_str)
        
        if matches:
            # Find the start index of the first match and the end index of the last match
            start_index = json_str.find(matches[0])
            end_index = json_str.rfind(matches[-1]) + len(matches[-1])
            
            # Insert "{" before the first match and "}" after the last match
            json_str = json_str[:start_index] + "{\n" + json_str[start_index:end_index] + "\n}" + json_str[end_index:]
        
            if verbose:
                print(f'Add missing Braces for the JSON string. - Done.')

    
    # check if the json_str missed the closing Brace
    if json_str.count('{') > json_str.count('}'):
        ## find the end index of last key-value pair and add the closing brace
        pattern = r'"[^"]+"\s*:\s*".+?"'
        # Find all matches of the pattern
        matches = re.findall(pattern, json_str)
        if matches:
            end_index = json_str.rfind(matches[-1]) + len(matches[-1])
            json_str = json_str[:end_index] + "\n}" + json_str[end_index:]
        else:
            json_str += '\n}' * (json_str.count('{') - json_str.count('}'))
        if verbose:
            print(f'Add missing closing Brace after the last KV. - Done.')
    # check if the json_str missed the closing Brace
    # if json_str.count('{') > json_str.count('}'):
    #     json_str += '\n}' * (json_str.count('{') - json_str.count('}'))
    #     if verbose:
    #         print(f'Add missing closing Brace in the end. - Done.')

    # Remove contents outside paired Braces.
    match1 = re.search(r".+(\{.*?\}).*", json_str, re.DOTALL)
    match2 = re.search(r"(\{.*?\}).+", json_str, re.DOTALL)
    if match1:
        json_str = match1.group(1)
        if verbose:
            print(f'Remove contents outside paired Braces. - Done.')
    elif match2:
        json_str = match2.group(1)
        if verbose:
            print(f'Remove contents outside paired Braces. - Done.')

    # Replace smart double quotes with straight ones
    if re.search(r'“|”', json_str):
        json_str = json_str.replace('“', '"').replace('”', '"') # curved to straight quote
        if verbose:
            print(f'Replace smart double quotes with straight ones. - Done.')
    
    # Replace \" to \'
    pattern = r'\\"'
    replacement = r"\\'"
    json_str = re.sub(pattern, replacement, json_str)

    # Remove back slash signs, if exist
    if re.search(r'\\', json_str):
        json_str = json_str.replace('\\', '')
        if verbose:
            print(f'Remove back slash signs, if exist. - Done.')
    
    # Drop extra comma befor closing Brace
    if re.search(r'",(\s*\n\s*})', json_str):
        json_str = re.sub(r'",(\s*\n\s*})', r'"\1', json_str)
        if verbose:
            print(f'Drop extra comma befor closing Brace. - Done.')

    # Add missing comma between key-value pairs
    if re.search(r'"(\s*\n\s*)"', json_str):
        json_str = re.sub(r'"(\s*\n\s*)"', r'": ",\1"', json_str)
        if verbose:
            print(f'Add missing comma between key-value pairs. - Done.')
       
    # Add double quotes to a string value, if missing
    if re.search(r'":\s*([^"01\s].*?)(?=$|\}|(\s*\n})|(,\s*\n\s*"))', json_str):
        json_str = re.sub(r'":\s*([^"01\s].*?)(?=$|\}|(\s*\n})|(,\s*\n\s*"))', r'": "\1"', json_str)
        if verbose:
            print(f'Add double quotes to a string value, if missing. - Done.')

    # add missing closing double quote at the end of the string value of the last key
    if re.search(r'":\s*"(.*?[^"])(?=(\n\s*\}))', json_str):
        json_str = re.sub(r'":\s*"(.*?[^"])(?=(\n\s*\}))', r'": "\1"', json_str)
        if verbose:
            print(f"Add the missing double quote to the string end. - Done.")
     
    # Replace double quotes within single quoted strings with =
    if re.search(r"'.*\".*'", json_str):
        json_str = re.sub(r"'(.*?)'", lambda m: "'" + m.group(1).replace('"', '=') + "'", json_str)
        if verbose:
            print(f"Replace double quotes within single quoted strings with =. - Done.")

    # Replace double quotes within a string value with single quotes
    if re.search(r'":\s*"(.*".*)(?=("(,\s*\n)|(\s*\n\})))', json_str):
        json_str = re.sub(r'":\s*"(.*)"', lambda m: '": "' + m.group(1).replace('"', "'") + '"', json_str)
        if verbose:
            print(f"Replace double quotes within a string value with single quotes. - Done.")
    
    # # some more ...
    
    # Try to load the JSON string
    try:
        json.loads(json_str)
        if verbose:
            print('json fixed:-)', end='', flush=True)
        return json_str
    except json.JSONDecodeError as e:
        # print(json_string)
        print(f"fix_json: failed. Error: {e}")
        if verbose:
            print(f"Revised but failed json string: \n{json_str}")
        return None


