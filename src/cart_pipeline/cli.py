from __future__ import annotations
import argparse
import json
from .pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser(description='Adaptive surrogate-assisted CAR-T study pipeline')
    p.add_argument('--config', required=True, help='YAML configuration file')
    p.add_argument('--output', required=True, help='Output directory')
    args = p.parse_args()
    summary = run_pipeline(args.config, args.output)
    print('\n=== Study summary ===')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
