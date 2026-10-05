from __future__ import annotations
import os
import platform
import shutil
import subprocess
from pathlib import Path

def source_dir() -> Path:
    return Path(__file__).resolve().parent / 'cpp'

def default_binary_path() -> Path:
    name = 'cart_solver.exe' if os.name == 'nt' else 'cart_solver'
    return Path(__file__).resolve().parent / 'bin' / name

def build_solver(force: bool=False, verbose: bool=True) -> Path:
    src = source_dir() / 'cart_solver.cpp'
    out = default_binary_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and (not force) and (out.stat().st_mtime >= src.stat().st_mtime):
        return out
    system = platform.system().lower()
    if system != 'windows':
        compiler = shutil.which('g++') or shutil.which('clang++')
        if compiler:
            cmd = [compiler, '-O3', '-std=c++17', str(src), '-o', str(out)]
            omp_cmd = cmd[:-2] + ['-fopenmp'] + cmd[-2:]
            try:
                if verbose:
                    print('[build]', ' '.join(omp_cmd))
                subprocess.run(omp_cmd, check=True)
                return out
            except subprocess.CalledProcessError:
                if verbose:
                    print('[build] OpenMP build failed; retrying without OpenMP.')
                subprocess.run(cmd, check=True)
                return out
    cmake = shutil.which('cmake')
    if cmake:
        build = source_dir() / '_build'
        build.mkdir(exist_ok=True)
        cfg = 'Release'
        subprocess.run([cmake, '-S', str(source_dir()), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release'], check=True)
        subprocess.run([cmake, '--build', str(build), '--config', cfg], check=True)
        candidates = list(build.rglob('cart_solver.exe' if os.name == 'nt' else 'cart_solver'))
        if not candidates:
            raise RuntimeError('CMake completed but cart_solver binary was not found.')
        shutil.copy2(candidates[0], out)
        return out
    raise RuntimeError('No C++17 compiler was found. Install g++/clang++ (Linux/macOS) or Visual Studio Build Tools + CMake (Windows), then run cart-build-solver.')

def main() -> None:
    p = build_solver(force=True)
    print(p)
if __name__ == '__main__':
    main()
