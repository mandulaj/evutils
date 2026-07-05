#!/usr/bin/env python3
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Generate markdown benchmark tables from pytest-benchmark JSON")
    parser.add_argument("--json", type=str, default="bench.json", help="Path to pytest-benchmark JSON file. If missing, it will run pytest to generate it.")
    args = parser.parse_args()

    json_path = Path(args.json)

    if not json_path.exists():
        print(f"File {json_path} not found. Running benchmarks to generate it...")
        # Run read and write benchmarks on the small dataset
        cmd = ["pytest", "benchmarks/", "-k", "test_read or test_write", "--benchmark-json", str(json_path)]
        subprocess.run(cmd, check=False)

    if not json_path.exists():
        print("Failed to generate benchmark JSON.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        
    read_results = defaultdict(dict)
    write_results = defaultdict(dict)
    
    read_formats = set()
    write_formats = set()

    for bench in data.get("benchmarks", []):
        name = bench["name"]
        
        # Parse read benchmarks
        if name.startswith("test_read_evutils["):
            fmt = name.split("[")[1].split("]")[0]
            read_results["evutils"][fmt] = bench["stats"]["mean"]
            read_formats.add(fmt)
        elif name.startswith("test_read_compare["):
            param = name.split("[")[1].split("]")[0]
            if "-" in param:
                lib, fmt = param.split("-")
                read_results[lib][fmt] = bench["stats"]["mean"]
                read_formats.add(fmt)
                
        # Parse write benchmarks
        elif name.startswith("test_write_evutils["):
            fmt = name.split("[")[1].split("]")[0]
            write_results["evutils"][fmt] = bench["stats"]["mean"]
            write_formats.add(fmt)
        elif name.startswith("test_write_expelliarmus["):
            fmt = name.split("[")[1].split("]")[0]
            write_results["expelliarmus"][fmt] = bench["stats"]["mean"]
            write_formats.add(fmt)
            
    def generate_table(results, formats, title):
        if not results:
            return ""
        
        formats = sorted(list(formats))
        out = f"### {title}\n\n"
        out += "| Library | " + " | ".join([f"{fmt.upper()}" for fmt in formats]) + " |\n"
        out += "|---|" + "|".join(["---" for _ in formats]) + "|\n"
        
        # Ensure evutils is always the first row
        libs = sorted(results.keys(), key=lambda x: (x != "evutils", x))
        
        for lib in libs:
            row = f"| **{lib}** | "
            cols = []
            for fmt in formats:
                val = results[lib].get(fmt)
                if val is not None:
                    cols.append(f"{val:.3f} s")
                else:
                    cols.append("N/A")
            row += " | ".join(cols) + " |\n"
            out += row
        
        return out
        
    markdown_output = "## Benchmark Comparison (Mean Time in Seconds)\n\n"
    markdown_output += generate_table(read_results, read_formats, "Reading") + "\n"
    markdown_output += generate_table(write_results, write_formats, "Writing") + "\n"
    
    # Extract hardware info if available
    machine_info = data.get("machine_info", {})
    if machine_info:
        cpu = machine_info.get("cpu", {}).get("brand_raw", "Unknown CPU")
        system = machine_info.get("system", "Unknown OS")
        release = machine_info.get("release", "")
        py_ver = machine_info.get("python_version", "")
        markdown_output += f"**Hardware:** {cpu} | **OS:** {system} {release} | **Python:** {py_ver}\n\n"
        
    markdown_output += "*Lower is better. Generated dynamically by `scripts/generate_benchmark_table.py`.*\n"

    print(markdown_output)
    
    # Attempt to inject into README.md
    readme_path = Path("benchmarks/README.md")
    if readme_path.exists():
        content = readme_path.read_text()
        start_marker = "## Benchmark Comparison"
        end_marker = "> **Note**: `tonic` is intentionally not included."
        
        if start_marker in content and end_marker in content:
            before = content.split(start_marker)[0]
            after = content.split(end_marker)[1]
            new_content = before + markdown_output + "\n" + end_marker + after
            readme_path.write_text(new_content)
            print("Successfully updated benchmarks/README.md!")

if __name__ == "__main__":
    main()
