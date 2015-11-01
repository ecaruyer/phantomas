Phantomas
=========
Phantomas (http://www.emmanuelcaruyer.com/phantomas.php) is an open-source
software library for the creation of realistic phantoms in diffusion MRI.
This is intented as a tool for the validation of methods in acquisition,
signal and image processing, local reconstruction and fiber tracking in
diffusion MRI. Phantomas is released under the terms of the revised BSD
license. You should have received a copy of the license together with the
software, please refer to the file ''LICENSE''.


Dependencies
------------
Phantomas is written in Python with parts in C. You will need Python 2.7,
and Cython installed on your computer.

Besides, Phantomas depends on:
- numpy, 
- scipy, 
- scikits-sparse

Optionally, you may also need:
- vtk,
- matplotlib.

Dependencies can be satisfied by running `pip install -r requirements.txt`

On ubuntu 14.04, installation of scikits-parse by pip
(`pip install scikits.parse`) failed with the following line:
`scikits/sparse/cholmod.c:245:33: fatal error: suitesparse/cholmod.h: No such file or directory`
The issue was solved installing the package 'libsuitesparse-dev':
`apt-get install libsuitesparse-dev`

Build instructions
------------------
Under linux, simply run (as root) 

# python setup.py install


Getting started
---------------
Phantomas provides a number of scripts, and examples. The main script is
phantomas_dwis. For more info, type:

$ phantomas_dwis -h

