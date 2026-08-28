"""Verify that a directory reorganisation actually landed in the code.

Running a script end to end does NOT verify a path migration. If a write still
points at the old location, the write still *succeeds* — it just creates the
file in the old place. Nothing raises, the script prints its summary, and the
moved copy in `results/` quietly goes stale while a fresh duplicate appears
next to the source. The failure is invisible to "did it run?".

This checks it statically instead: every write of a result-type file should
resolve under the results/models directory, not the script directory.

Usage:
    python check_paths.py                 # report
    python check_paths.py --strict        # exit 1 if anything is unmigrated

Zero findings does not by itself prove the reorg is complete — it proves no
script writes a result next to its source. Pair it with the runtime check at
the bottom of this docstring.

Runtime check, which catches what static analysis cannot (paths built at run
time, or by libraries):

    git status --porcelain tacticai_task2/ \\
        | grep -E '^\\?\\? tacticai_task2/[^/]+\\.(csv|png|pt)$'

Run that AFTER executing a script. Any result file showing up as untracked in
the script directory is a write that was not migrated. Two details matter: the
`^??` anchor, without which the pattern also matches the `R  old -> new` rename
lines the migration itself produces (findings that are not findings), and the
`[^/]+`, which keeps a correctly-placed write under results/ from matching.
`parquet` is not in the list: derived data deliberately stays in the script
directory, so a parquet there is expected, not a miss.
"""
import ast
import os
import re
import sys

SP = os.path.dirname(os.path.abspath(__file__))

# Files that belong in the reorganised directories.
RESULT_EXT = ('.csv', '.png')
MODEL_EXT = ('.pt',)
# Derived data that stays put unless the reorg says otherwise.
DATA_EXT = ('.parquet',)

WRITE_CALLS = {'to_csv', 'to_parquet', 'savefig', 'save'}
# Directory names that count as "migrated".
MIGRATED = ('results', 'models', 'figures')


def path_strings(node):
    """Recover a literal or f-string path argument, with {vars} as markers."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(v.value)
            else:
                inner = getattr(v, 'value', None)
                out.append('{%s}' % (getattr(inner, 'id', None)
                                     or getattr(inner, 'attr', '?')))
        return ''.join(out)
    return None


def scan(path):
    """Yield (lineno, call, path_expr, verdict) for every write in one file."""
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError as e:
        yield (0, 'PARSE ERROR', str(e), 'error')
        return

    # resolve simple module-level string bindings, e.g. cache = f'{SP}/x.parquet'
    bound = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Name):
                s = path_strings(n.value)
                if s:
                    bound[t.id] = s

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = getattr(n.func, 'attr', None)
        if fn not in WRITE_CALLS or not n.args:
            continue
        for arg in n.args[:2]:
            s = path_strings(arg)
            if s is None and isinstance(arg, ast.Name):
                s = bound.get(arg.id)
            if not s:
                continue
            ext = os.path.splitext(s)[1].lower()
            if ext not in RESULT_EXT + MODEL_EXT + DATA_EXT:
                continue
            parts = s.split('/')
            in_migrated = any(p in MIGRATED for p in parts[:-1])
            if ext in DATA_EXT:
                verdict = 'data (expected to stay)'
            elif in_migrated:
                verdict = 'OK'
            else:
                verdict = 'UNMIGRATED'
            yield (n.lineno, fn, s, verdict)
            break


def main():
    strict = '--strict' in sys.argv
    bad = []
    print(f'{"script":<28}{"line":>5}  {"path":<46}verdict')
    print('-' * 96)
    for fn in sorted(f for f in os.listdir(SP) if f.endswith('.py')):
        if fn == os.path.basename(__file__):
            continue
        for lineno, call, s, verdict in scan(os.path.join(SP, fn)):
            if verdict == 'data (expected to stay)':
                continue
            print(f'{fn:<28}{lineno:>5}  {s:<46}{verdict}')
            if verdict == 'UNMIGRATED':
                bad.append((fn, lineno, s))

    print()
    if bad:
        print(f'{len(bad)} unmigrated write(s) in '
              f'{len(set(b[0] for b in bad))} script(s):')
        for fn, lineno, s in bad:
            print(f'  {fn}:{lineno}  ->  {s}')
        print('\nEach of these still writes next to the source file. The script')
        print('will run without error and the copy under results/ will go stale.')
    else:
        print('No unmigrated writes found.')
        print('Also run the runtime check in this file\'s docstring — static')
        print('analysis cannot see paths assembled at run time.')

    if strict and bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
