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

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

The AIDA interpreter is a stan
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install aida package
   ```sh
   python -m pip install -e /path/to/aida
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

```py
import aida
import numpy as np
import matplotlib.pyplot as plt
```

```py
glat = np.linspace(-90.0, 90)
glon = np.linspace(-180.0, 180.0, 70)
```

```py
Model = aida.AIDAState()
Model.readFile("./tests/data/output_3_231201_042500.h5")
```

```py
Output = Model.calc(
    lat=glat, lon=glon, grid="3D", TEC=True, MUF3000=True, collapse_particles=True
)
```

```py
BkgModel = Model.background()
BkgOutput = BkgModel.calc(
    lat=glat, lon=glon, grid="3D", TEC=True, MUF3000=True, collapse_particles=True
)
```

```py
for i, d in enumerate(['NmF2', 'hmF2', 'TEC', 'MUF3000']):
     
    if i % 4 == 0:
        fig, axs = plt.subplots(3, 4, squeeze=False)
        fig.set_size_inches(12, 9)
    ix = i % 4

    pcm = axs[0, ix].pcolor(Output.glon, Output.glat, Output[d].T)
    axs[0, ix].set_title(f"AIDA {d}")
    fig.colorbar(pcm, ax=axs[0, ix])
    axs[1, ix].pcolor(BkgOutput.glon, BkgOutput.glat, BkgOutput[d].T)
    axs[1, ix].set_title(f"Background {d}")
    fig.colorbar(pcm, ax=axs[1, ix])
    pcm = axs[2, ix].pcolor(
        BkgOutput.glon, BkgOutput.glat, (Output[d] - BkgOutput[d]).T
    )
    axs[2, ix].set_title(f"Difference {d}")
    fig.colorbar(pcm, ax=axs[2, ix])

```

[![Product Name Screen Shot][product-screenshot]](https://example.com)

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


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[product-screenshot]: tests/data/output.png