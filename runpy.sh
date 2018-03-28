#!/bin/bash 
# takes arg of python file (e.g. myscript.py) and runs it in a screen 
# quits screen when done 
screen -d -m -S $(echo $1 | cut -f 1 -d '.') python $1; 
screen -X $(echo $1 | cut -f 1 -d '.');