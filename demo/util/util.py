import re

def remove_irrelevant_text(txt):
    """ Remove misleading but current trace irrelevant text"""
    # remove irrelevant heading
    new_txt = re.sub(r"Clinical indication for E(K|C)G:[\w\d\s\-\.\(\),;]*(?=(\n *\n\w))", '', txt)
    # remove irrelevant ending
    new_txt = re.sub(re.compile(r"Please see final interpretation.*?$", re.DOTALL), '', new_txt)    
    return new_txt.strip()
