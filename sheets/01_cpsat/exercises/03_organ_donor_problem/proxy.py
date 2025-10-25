import runpy, sys

sys.argv = ["visualization.py", "--basic"]
runpy.run_path("visualization.py", run_name="__main__")