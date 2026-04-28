with open("dinoplay/app.py", "r") as f:
    code = f.read()
code = code.replace("import time", "import time\n                print('do_live_search evaluated', active, flush=True)")
with open("dinoplay/app.py", "w") as f:
    f.write(code)
