#!/usr/bin/env python3
import subprocess, pathlib, sys

paths = subprocess.check_output(
    ["git", "ls-files", "-z"], text=True
).split("\0")

print(paths)

# 倒序，先文件再目录
for p in sorted(filter(None, paths), reverse=True):
    if "excuter" in p:
        print(f"find: {p}\n")
        new = p.replace("excuter", "executor")

        subprocess.run(["git", "mv", p, new], check=True)

print("Done. Now run your tests.")
