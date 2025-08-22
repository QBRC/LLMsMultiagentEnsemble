import os
import sys

for p in ['ensemble_llms_app','llm_app','llm_agent','fix_json','evaluate']:
    package_dir = os.path.abspath(f"../{p}")
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
