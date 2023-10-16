call ./__local__/setpaths.cmd
set MYPYPATH=./__local__;%xtuples%;%xtenors%;
python -m mypy .%1 --check-untyped-defs --soft-error-limit=-1