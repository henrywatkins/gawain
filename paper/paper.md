---
title: 'Gawain: A Python package for 2D magnetohydrodynamics'
tags:
  - Python
  - mhd
  - plasma physics
authors:
  - name: Henry Watkins^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0001-6330-6195
    affiliation: 1
affiliations:
 - name: Department of Physics and Astronomy, Imperial College London
   index: 1
date: 6 March 2021
bibliography: paper.bib
---

## Gawain: A Python Package for 2D Magnetohydrodynamics

### Summary

Gawain is a Python package designed for simulating two-dimensional magnetohydrodynamic (MHD) systems. It provides a framework for modeling both inviscid, compressible hydrodynamics and ideal MHD in two dimensions. The package is tailored for simplicity and ease of use, making it accessible to researchers and students in the field of plasma physics. Gawain leverages Python's capabilities to offer a flexible simulation environment with minimal dependencies, ensuring that users can easily set up and run simulations without extensive software overhead.

### Statement of Need

The study of magnetohydrodynamics is crucial for understanding various physical phenomena in astrophysics, fusion research, and space physics. However, existing MHD simulation tools can be complex and challenging to use, especially for those new to computational physics. Gawain addresses this gap by providing a straightforward, Python-based solution that allows users to perform MHD simulations with ease. Its simplicity does not compromise its functionality, as it includes essential features such as different boundary conditions (fixed-value, reflective, periodic, outflow) and the ability to simulate the effects of gravitational fields or arbitrary source functions on fluid dynamics. This makes Gawain an invaluable tool for educational purposes and preliminary research investigations where ease of use and flexibility are paramount.

### Features

- **Simulation Capabilities**: Supports 2D inviscid, compressible hydrodynamics and ideal MHD.
- **Boundary Conditions**: Offers fixed-value, reflective, periodic, and outflow boundary conditions.
- **Flexibility**: Allows for simulations with gravitational fields or arbitrary source functions.
- **Integration Methods**: Includes several integrators such as Euler, Lax-Wendroff, Lax-Friedrichs, Van Leer, and HLL flux solvers.
- **Minimal Dependencies**: Designed to be lightweight with few external library requirements.

### Getting Started

To begin using Gawain, users can install the package via the standard Python setup process. Example scripts and Jupyter notebooks are provided in the documentation directory to guide users through creating and running simulations. Users can configure their simulations by setting parameters in a Python dictionary and passing them to the `rungawain` function. This approach ensures that even those with limited programming experience can effectively utilize the software for their research needs.

### Conclusion

Gawain is a powerful yet user-friendly tool that democratizes access to MHD simulations. Its focus on simplicity makes it an excellent choice for educational settings and initial research explorations in plasma physics. By reducing the complexity typically associated with MHD simulation software, Gawain enables a broader audience to engage with advanced computational modeling techniques.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_3e2af07f-d831-482e-a8b4-84c459a2dc9e/1f80d06d-68ca-45fe-a478-e107c64f351f/paper.md
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_3e2af07f-d831-482e-a8b4-84c459a2dc9e/f01d9e5c-1651-4be1-bd7c-34ae6ffbc2c5/README.md