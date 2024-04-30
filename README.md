# Black Hole Superradiance for Ultralight Bosons

<em><font size="4">A Python package and data repo for calculating BHSR rates and constraining the properties of ultralight bosons.</font></em>

<table>
   <tbody>
      <tr><th scope="row">Developer(s)</th><td>Sebastian Hoof</td></tr>
      <tr><th scope="row"> Maintainer(s)</th><td><a href = "mailto:s.hoof.physics@gmail.com">Sebastian Hoof</a></td></tr>
      <tr><th scope="row">Licence</th><td>BSD 3-clause licence, see <a href="LICENSE">the licence file</a> for details</td></tr>
   </tbody>
</table>

## Results

Details about our proposed inference framework, which directly uses black hole mass and spin posterior samples to contrain ultralight bosons, are currently available on the arXiv as a preprint [arXiv:2405.xxxxx].
More details to follow after the peer review process.


## Installation


### Requirements

-  Python interpreter v3.10 (or higher)
-  Python packages: iminuit, numpy, numba, scipy, superrad, qnm

### Step-by-step guide

1. Install the dependencies into your Python environment via `python -m pip install iminuit numpy numba scipy superrad qnm`
2. Clone this repo via `git clone https://github.com/sebhoof/bhsr`


## Get stared
We include the simple Jupyter notebook [examples.ipynb](examples.ipynb) to demonstrate a few of the capabilities of our code.


## How to cite

Even if you wish to *only* cite our code, we still ask you to cite [[arXiv:2405.xxxxx]](https://arxiv.org/abs/2405.xxxxx) and to include a link to this Github repo.
Suffice to say that, sadly, code citations are still not as recoginsed as paper citations.

Depending on what parts of the code or repository you use, more works have to be acknowledged.
We use BH data and results, and (re-)distribute posterior samples from BH data.

| Black hole name | Samples | Reference(s) |
| :--- | :--- | :--- |
| IRAS 09149-6206 | no | [[arXiv:1705.02345]](https://arxiv.org/abs/1705.02345), [[arXiv:2009.08463]](https://arxiv.org/abs/2009.08463), [[arXiv:2009.10734]](https://arxiv.org/abs/2009.08463) |
| M33 X-7 | yes | [[arXiv:0803.1834]](https://arxiv.org/abs/0803.1834) |
