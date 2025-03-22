<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://spaceweather.bham.ac.uk/">
    <img src="https://spaceweather.bham.ac.uk/static/images/serene-logo.22a05ba05f53.png" alt="Logo" height="80">
  </a>

  <h3 align="center">AIDA: Advanced Ionospheric Data Assimilation</h3>

  <p align="center">
    <a href="https://spaceweather.bham.ac.uk/output/"><strong>AIDA Real-Time Data Model Output»</strong></a>
    <br />
    <a href="https://spaceweather.bham.ac.uk/output/">
    <img src="https://spaceweather.bham.ac.uk/output/aida/" alt="AIDA Output" height="220">
  </a>
    <br />
    <a href="https://gitlab.bham.ac.uk/elvidgsm-dasp/aida-ionosphere/-/issues">Report Bug</a>
    ·
    <a href="https://gitlab.bham.ac.uk/elvidgsm-dasp/aida-ionosphere/-/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

AIDA is a real-time ionosphere/plasmasphere data assimilation model. AIDA uses measurements from ground- and satellite-based Global Navigation Satellite (GNSS) receivers and ionosondes to produce an improved global ionospheric representation. This package contains the necessary software to read the output files produced by the AIDA system, and produce outputs of electron density, Total Electron Content (TEC), MUF3000, and various ionospheric profile parameters (NmF2, foF2, hmF2, etc.)

The AIDA interpreter is a standalone package to read the AIDA output files. It requires output from the AIDA model from [https://spaceweather.bham.ac.uk/output/](https://spaceweather.bham.ac.uk/output/).

To access AIDA model output and forecast products, it is necessary to create a free account at [https://spaceweather.bham.ac.uk/accounts/register/](https://spaceweather.bham.ac.uk/accounts/register/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites



### Installation (Local Computer)

_Currently the AIDA interpreter must be installed using git. In the future the model interpreter will be made available on PyPI._

1. Clone the repo
   ```sh
   git clone https://gitlab.bham.ac.uk/elvidgsm-dasp/aida-ionosphere.git
   ```
2. Install aida package
   ```sh
   python -m pip install -e /path/to/aida
   ```

  (In Ubuntu, the openmpi dependency will require installation first)

  ```sh
  sudo apt install libopenmpi-dev
   ```


### Installation (BlueBEAR)

_The AIDA interpreter is already available on BlueBEAR through the `dasp` project virtual environment._

```sh
$ module purge; module load bluebear
$ module load Python/3.9.5-GCCcore-10.3.0-bare
$ source /rds/projects/e/elvidgsm-dasp/DASP-website/code/venv/bin/activate 
```

_You can check if the package is available using the command `pip freeze`, which will print all currently installed python packages._

```sh
$ pip freeze | grep aida
  ionodasp @ file:///rds/projects/e/elvidgsm-dasp/DASP-website/code/ dasp_assimilation
```



### Finding AIDA Output Files on BlueBEAR

Output from the online AIDA models can be found at `/rds/projects/e/elvidgsm-dasp/DASP-website/AIDA_output/archive/{model}/{yyyy}/{doy}/`.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

The following python packages will be used in the demonstration.

```py
import aida
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
```

### AIDA State Object

All AIDA calculations are performed through the `AIDAState` class.

```py
# Create an empty AIDAState
Model = aida.AIDAState()
```

An AIDA State must be populated with data from an AIDA output file using the `AIDAState.readFile()` method. 

```py
# load test output file
Model.readFile("./tests/data/output_3_231201_042500.h5")
```

The `AIDAState` class has several attributes:

* `AIDAState.Version` is the version and operating mode of the AIDA model which produced the file
* `AIDAState.Time` is the time for which this file provides a description of the ionospheric state (in UNIX Epoch format)
* `AIDAState.Metadata` contains information about the model used to generate the file
    * `AIDAState.Metadata.ForecastStart` gives the time of the previous AIDA output used to forecast this output (in UNIX Epoch format)
    * `AIDAState.Metadata.NeQuickFlux` gives the F10.7 cm flux passed to NEQuick for generating the AIDA background model
    * `AIDAState.Metadata.NeQuickVersion` is the version of NeQuick used for generating the AIDA background model
    * `AIDAState.Metadata.InstrumentsGNSS` gives a list of GNSS receivers used for generating the output
    * `AIDAState.Metadata.InstrumentsIonosonde` gives a list of ionosondes used for generating the output

The background state of an AIDA model output can be accessed using the `AIDAState.background()` method, which returns a copy of the `AIDAState` object with all parameters set to match the background state.

### Example 1: Electron Density (3D Grid)

For most cases, the `AIDAState.calc()` method is used to calculate values of interest. `AIDAState.calc()` accepts `glat`, `glon`, and `alt` arguments, and the shape of the output is determined by the `grid` argument. If `grid = '3D'`, the `glat`, `glon`, and `alt` inputs will be broadcast together to produce a regular 3D grid. When `grid=3D` or `grid=2D`, AIDA can be more efficient when performing the internal spherical harmonic transformations, which is much more efficient for large, regular grids. When calculating arbitrary points, `grid=1D` must be used.

```py
# create several arrays of coordinates
glat = np.linspace(-90.0, 90, 100)
glon = [0.0, 180.0]
alt = np.linspace(0.0, 2e3, 150)

# calculate AIDA output
Output = Model.calc(lat=glat, lon=glon, alt=alt, grid="3D", collapse_particles=True)
```

By default, `Output` is in the form of an `xarray.Dataset` object, for more information see [the `xarray` documentation](https://docs.xarray.dev/en/stable/index.html). To get the output in the form a python `dict`, pass `as_dict=True` to `AIDAState.calc()`.

The output of `AIDAState.calc()` will have a field `Ne` containing the electron density, along with fields for all of the AIDA ionospheric profile parameters. In a 3D grid, the `Ne` field will have a size `(glon, glat, alt)`, and all other parameters will have size `(glon, glat)`.

By default, the output fields will also have a dimension labelled `particle`, which corresponds to the ensemble size of the AIDA state. When working with model output, the particle dimension will always be of size 1, and so this extra dimension can be suppressed with the argument `collapse_particles=True`.

```py
# create figure
fig, axs = plt.subplots(len(glon), 1, squeeze=False)
fig.set_size_inches(12, 9)

for i, lon in enumerate(glon):
    # plot latitudinal section for each longitude
    ax = axs[i, 0]
    pcm = ax.pcolor(Output.glat, Output.alt, np.log10(Output.Ne.sel(glon=lon).T))
    ax.set_title(f"AIDA Ne: lon = {lon}")
    fig.colorbar(pcm, ax=ax, label=f"log10 {Output['Ne'].attrs['units']}")
    pcm.set_clim(8, 12.5)
    pcm.set_cmap(colormaps["magma"])

```

<img src="tests/data/output_3D.png" alt="Example 3D Output" height="380">


### Example 2: Electron Density (2D Grid)

For irregular grids, the argument `grid=2D` can be used. This requires `glat` and `glon` to be the same size.

```py
# create irregular grid
glat = [[45.0, 65.0, 77.0], [30.0, 40.0, 50.0]]
glon = [[0.0, 345.0, -160.0], [-60.0, 61.1, 62.2]]
alt = np.linspace(90.0, 500.0, 150)

Output = Model.calc(
    lat=glat, lon=glon, alt=alt, grid="2D", collapse_particles=True, as_dict=True
)
```

In this example, the output is in `dict` format.

```py
# create figure
fig, axs = plt.subplots(1, 2, squeeze=False)
fig.set_size_inches(6, 6)

for i, lon in enumerate(glon):
    # plot profiles and mark layers
    ax = axs[0, i]
    ax.plot(Output["Ne"][i, :].T, Output["alt"])
    ax.scatter(Output["NmF2"][i, :], Output["hmF2"][i, :])
    ax.scatter(Output["NmF1"][i, :], Output["hmF1"][i, :])
    ax.scatter(Output["NmE"][i, :], Output["hmE"][i, :])
```

<img src="tests/data/output_2D.png" alt="Example 2D Output" height="380">

### Example 3: Maps

AIDA uses a parameterized ionospheric profile, and many of these profile parameters are of interest, in addition to the electron density. `AIDAState.calc()` will generate these values along with the electron density. If the `alt` argument is omitted, electron density `Ne` will not be calculated. 

AIDA can also calculate derived quantities, such as the Total Electron Content (TEC) and MUF3000. These are more computationally expensive, and are disabled by default. They can be included by passing the arguments `TEC=True` and `MUF3000=True` to `AIDAState.calc()`.

```py
glat = np.linspace(-90.0, 90)
glon = np.linspace(-180.0, 180.0, 70)

Output = Model.calc(
    lat=glat, lon=glon, grid="3D", TEC=True, MUF3000=True, collapse_particles=True
)
```

In this example, the AIDA model will be compared to the values given in the background model.

```py
BkgModel = Model.background()
BkgOutput = BkgModel.calc(
    lat=glat, lon=glon, grid="3D", TEC=True, MUF3000=True, collapse_particles=True
)
```

```py
fig, axs = plt.subplots(3, 5, squeeze=False)
fig.set_size_inches(15, 10)

for i, d in enumerate(["NmF2", "hmF2", "TEC", "MUF3000", "foF2"]):

    ax = axs[0, i]
    pcm = ax.pcolor(Output.glon, Output.glat, Output[d].T)
    ax.set_title(f"AIDA {d}")
    fig.colorbar(pcm, ax=ax, label=Output[d].attrs["units"])
    pcm.set_cmap(colormaps["magma"])
    cl = pcm.get_clim()

    ax = axs[1, i]
    pcm = ax.pcolor(BkgOutput.glon, BkgOutput.glat, BkgOutput[d].T)
    ax.set_title(f"Background {d}")
    fig.colorbar(pcm, ax=ax, label=Output[d].attrs["units"])
    pcm.set_cmap(colormaps["magma"])
    pcm.set_clim(cl)

    ax = axs[2, i]
    pcm = ax.pcolor(BkgOutput.glon, BkgOutput.glat, (Output[d] - BkgOutput[d]).T)
    ax.set_title(f"Difference {d}")
    fig.colorbar(pcm, ax=ax, label=Output[d].attrs["units"])
    cl = np.max(np.abs(pcm.get_clim()))
    pcm.set_clim(-cl, cl)
    pcm.set_cmap(colormaps["bwr"])

```

 <img src="tests/data/output_Map.png" alt="Example Output Map" height="380">

### Example 4: Electron Density (Advanced Usage)

For some performance-critical applications, it is desireable to calculate only the electron density without additional overhead. This can be achieved with the `AIDAState.calcNe()` method. Like `AIDAState.calc()`, it accepts `glat`, `glon`, `alt`, and `grid`, but outputs only the electron density `Ne` as a `numpy` array. `AIDAState.calcNe()` is often 2 to 5 times faster than calling `AIDAState.calc()` for grid points with 1e3 to 1e4 unique lat/lon pairs.

Use of `AIDAState.calcNe()` is not recommended unless required, such as when calculating sample points for sTEC calculations. 

```py
def cartsph(x, y, z):
    # convert from ECEF to GEO
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)

    return rad, lat, lon

def sphcart(rad, lat, lon):
    # convert from GEO to ECEF
    x = rad * np.cos(lat) * np.cos(lon)
    y = rad * np.cos(lat) * np.sin(lon)
    z = rad * np.sin(lat)

    return x, y, z
```

In this example slant Total Electron Content will be calculated between two points.

```py
# Earth's Radius
Re = 6371e3

# receiver and satellite coordinates (ECEF)
rX, rY, rZ = sphcart(Re, np.deg2rad(45.0), np.deg2rad(0.0))
tX, tY, tZ = sphcart(Re+22000e3, np.deg2rad(-30.0), np.deg2rad(10.0))

# create linear sample points
t = np.linspace(0,1,1000)
x = rX + (tX-rX) * t
y = rY + (tY-rY) * t
z = rZ + (tZ-rZ) * t

# find geo coordinates of sample points
rad, glat_r, glon_r = cartsph(x, y, z)
alt = (rad - Re) * 1e-3
glat = np.rad2deg(glat_r)
glon = np.rad2deg(glon_r)
```

The electron density can be calculated with:

```py
# output is in 1e11 m-3!
Ne = Model.calcNe(lat=glat, lon=glon, alt=alt, grid="1D")
```

```py
# linear distance (m)
d = np.sqrt((x - rX) ** 2 + (y - rY) ** 2 + (z - rZ) ** 2)

# integrate along ray
sTEC = np.trapz(Ne, d)  # array([3.36102813e+17])

```

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Will be distributed under a licence. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Benjamin Reid - [SERENE - University of Birmingham](https://spaceweather.bham.ac.uk/) - b.reid@bham.ac.uk

Project Link: [GitLab](https://gitlab.bham.ac.uk/elvidgsm-dasp/aida-ionosphere)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

