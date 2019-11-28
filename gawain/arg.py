import sys

print(sys.argv[0])
print(len(sys.argv))
print(str(sys.argv))

input_file = str(sys.argv[0])

sys.path.append(input_file)

preamble = """
   ______                     _
  / ____/___ __      ______ _(_)___
 / / __/ __ `/ | /| / / __ `/ / __ |
/ /_/ / /_/ /| |/ |/ / /_/ / / / / /
\____/\__,_/ |__/|__/\__,_/_/_/ /_/
-----------------------------------
"""

print(preamble)

value = 10
