# Machine learning project

This project was supposed to predict unknown pixels of an image. Project scored 26th out of 154 entires.
Loss was 16.45.
We had 30.000 samples, which were seperated as follows 18.000 for training, 6000 for validation and 6000 for test.

### Structure

Having a tree with the files and folders is nice to get an overview.
However, this is sometimes tedious to maintain and omitted.

```
|- architectures.py
|    Classes and functions for network architectures
|- datasets.py
|    Dataset classes and dataset helper functions
|- main.py
|    Main file. In this case also includes training and evaluation routines.
|- utils.py
|    Utility functions and classes. In this case contains a plotting function.
|- ex4.py
|     Helper function that is used for loading dataset.
|- prediction.py
|     Script that creates prediction based on model and input data.
|- working_config.json
|     An example configuration file. Can also be done via command line arguments to main.py.
```
