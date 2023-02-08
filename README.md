### How to use:

I included a setup.py so we can use

```
pip install -e .
```

to install the package into our virtual environment. This has the advantage that we dont get any import errors in Python when working with sibbling directories.

For example if you want to import a class called Resnet that lies in src/models/models.py you would import it like this

```py
from src.models.models import Resnet
```

This will work independent from where you execute the file or where the files lies on the hard drive.

In the config.py file in the root directory we can for example define paths as i did with ROOT_DIR and DATASET_DIR
These paths can then be used in any file like this:

```py
from config import ROOT_DIR
```

The paths will always be the right ones for each of our systems

### Data Preprocessing

I created a script for data processing which only keeps every 8th frame of the sequence.
If you did install the project as a pip package, you can just put the H36Pose folder i
shared on OneDrive into a folder datasets in the Final Project root and the script will automatically change all paths so they work on your system.
