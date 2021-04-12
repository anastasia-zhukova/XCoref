# Troubleshooting for the installation
Below is some troubleshooting information that may help if errors arise when following installation instructions from README.md 
(in our experience, installing newsalyze-backend on Linux or MacOS works just out-of-the-box, while Windows is a bit more tricky.

In some occasions, packagages might fail during the requirements installation. In the few known cases, install them manually (only if an error occurs):

```
pip install giveme5w1h
pip install stanfordnlp==0.1.2
```

Please note: On Windows, the torch package might not be installable, which results in the "ModuleNotFoundError: No module named 'tools.nnwrap'" error.
If you encounter this, you can run below command


Clone the repository, install required packages and resources.
```
pip install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl
git clone git@github.com:fhamborg/newsalyze-backend.git
pip install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-win_amd64.whl
cd newsalyze-backend
pip install -r requirements.txt
```

If you encounter the error elsewhere or on a different OS, follow the instructions given on https://pytorch.org/get-started/locally/.
