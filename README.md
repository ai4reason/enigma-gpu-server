# ENIGMA GPU Server

Tensorflow GPU server for fast evaluation with ENIGMA E Prover.

The server implementation is in the directory `tf-server`.  Example models are
in the directory `models`.  Update the values of variables `SERVER_IP` and
`SERVER_PORT` in the file `tf-server/tf_server_workers_threa.py` if the default
values (`127.0.0.1` and `8888`) need to be changed in your scenario.


## Requirements

The following Python packages are required (installable through `pip` or otherwise):

* `tensorflow` (1.5)
* `xopen`
* `asyncio`



