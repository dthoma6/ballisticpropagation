
# Install

```
git clone https://github.com/dthoma6/ballisticpropagation/tree/main
cd ballisticpropagation
pip install --editable .
```

# Description

This repository provides a script to ballistically propagate solar wind data.

It is based on Mailyan, B., C. Munteanu, and S. Haaland. "What is the best method to 
calculate the solar wind propagation delay?." Annales geophysicae. Vol. 26. 
No. 8. Copernicus GmbH, 2008.

The script follows Section 3.1 of that paper to ballistically propagate solar wind data 
from DSCOVR. Follow the instructions in ballistic_prop.py to propagate the solar
wind for the Gannon Storm.

The input files for the Gannon Storm can be found at:
```
http://mag.gmu.edu/git-data/dthoma6/ballisticpropagation/Gannon_input
```
The files are from May 2024 and were downloaded in late-June/early-July 2024. The 
output files resulting from this script, can be found at:
```
http://mag.gmu.edu/git-data/dthoma6/ballisticpropagation/Gannon_output
```


