# Black Hole Superradiance for Ultralight Bosons

<em><font size="4">A Python package and data repo for calculating BHSR rates and constraining the properties of ultralight bosons.</font></em>

<table>
   <tbody>
      <tr><th scope="row">Developer(s)</th><td>Sebastian Hoof</td></tr>
      <tr><th scope="row"> Maintainer(s)</th><td><a href = "mailto:s.hoof.physics@gmail.com">Sebastian Hoof</a></td></tr>
      <tr><th scope="row">License</th><td>BSD 3-clause license, see <a href="LICENSE">the licence file</a> for details</td></tr>
   </tbody>
</table>

## Results

Our proposed inference framework, direcly using black hole parameter samples, is currently available on the arXiv as a preprint: [arXiv:2405.xxxxx].


## Installation


### Requirements

-  Python interpreter v3.9 (or higher)
-  Python packages: iminuit, numpy, numba, scipy, superrad, qnm

### Step-by-step guide

1. Install the dependencies into your Python environment via `python -m pip install iminuit numpy numba scipy superrad qnm`
2. Clone this repo via `git clone https://github.com/sebhoof/bhsr`


## Get stared
We include the simple Jupyter notebook [examples.ipynb](examples.ipynb) to demonstrate a few of the capabilities of our code.


## How to cite

**All references mentioned below are can be found in the [references.bib](references.bib) file.**
You may also consider using the [BibCom tool](https://github.com/sebhoof/bibcom) to generate a list of references from the arXiv numbers or DOIs.

If you wish to *only* cite our code, we still ask you to cite [[arXiv:2405.xxxxx]](https://arxiv.org/abs/2405.xxxxx) and link to this Github repo.
Suffice to say that, sadly, paper citations are still viewed as more important than code citations.

Depending on what parts of the code or repository you use, more works have to be acknowledged.
We (re-)distribute posterior samples from BH data and external code, which need to be acknowledged accordingly.

| Black hole ID | Reference(s) |
| :--- | :--- |
| M33 X-7 | [[arXiv:0803.1834]](https://arxiv.org/abs/0803.1834) |
| IRAS 09149-6206 | To be added |