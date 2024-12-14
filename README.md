# LLE
LLE simulation for the summer research 2024.

Simulation of dissipative Kerr solitons with Lugiato-Lefever Equation.
Different from widely used LLE solvers, this code include the interplay of counter-clockwise WGMs.

The files should be structured like this:
```
Folder
├── LLE
│   ├── *.py
├── output
└── cache
```

Dependent modules: numpy, numba, ipython, ipywidgets, matplotlib, rocket-fft

The "rocket-fft" is not explicitly imported, but should be installed in the environment to support the coherent working of numpy and numba.

Interactive plots are tuned for VS Code.
