# ENIGMA GPU Server

Tensorflow GPU server for fast evaluation with ENIGMA E Prover.

## Running the Server

The server implementation is in the directory `tf-server`.  The main launch file is `tf-server/tf_server_workers_thread.py`.
Example models are in the directory `models`.  Update the values of variables `SERVER_IP` and
`SERVER_PORT` in the file `tf-server/tf_server_workers_thread.py` if the default
values (`127.0.0.1` and `8888`) need to be changed in your scenario.

## Running the E Client

Download and compile an E client with GPU server support from

* https://github.com/ai4reason/eprover/tree/enigma

To instruct the client to connect to the GPU server, you must use `EnigmaticTfs` clause weigth function in your E Prover strategy.  Its syntax is as follows:

```
EnigmaticTfs(prio_fun, server_ip, server_port, context_size, weight_type, threshold)
```

where

* `prio_fun` is E's standard priority function (we use `ConstPrio`),
* `server_ip` is server's IP address as a string in the format `nn.nn.nn.nn`,
* `server_port` is server's port at integer,
* `context_size` is the size of the context query,
* `weight_type` should be set to `2` (to use a sigmoid-like function to translate neural network's logits to the interval _(0,1)_)
* `threshold` is a threshold to decide which clause weights should be consider positives/negatives (use 0.5 as a default).

An example E strategy with `EnigmaticTfs` is as follows:

```
eprover -s --print-statistics --resources-info --definitional-cnf=24 --split-aggressive 
  --simul-paramod --forward-context-sr --destructive-er-aggressive --destructive-er
  --prefer-initial-clauses -tKBO6 -winvfreqrank -c1 -Ginvfreq -F1 --delete-bad-limit=150000000 
  -WSelectMaxLComplexAvoidPosPred --delayed-eval-cache=128 
  -H'(1*EnigmaticTfs(ConstPrio,127.0.0.1,8888,1024,2,0.5))' problem.p
```

Note that the argument `--delayed-eval-cache=128` is used to select the query size.

## Running the 2-phase ENIGMA: GNN with LGB pre-filtering

In order to use a fast and cheap pre-filtering of the clauses sent to the GPU server, use the same clause weight function `EnigmaticTfs` as above, but with additional parameters.

```
EnigmaticTfs(..., threshold, lgb_model_dir, lgb_weight_type, lgb_threshold)
```

The additional parameters are as follows:

* `lgb_model_dir` is the directory with LGB model,
* `lgb_weight_type` should be set to `1` (to use the threshold below) 
* `lgb_threshold` is the threshold to decide which clauses to evaluate on the GPU server (a real number between _0_ and _1_).

## Running E with parental guidance.

Download and compile an E client from

* https://github.com/zariuq/eprover/tree/parentalguidance_frozen

Parental guidance is used via a commandline argument to E:

```
--filter-generated-clauses=parental_lgb_model_dir
--filter-generated-threshold=parental_threshold
```

An LGB model for clause selection can be used via the clause weight function `EnigmaticLgb`:

```
EnigmaticLgb(prio_fun, selection_lgb_model_dir, weight_type, threshold)
```

where 

* `parental_lgb_model_dir` is the directory with the LGB model for parental guidance,
* `parental_threshold` is the threshold below which a pair of inference parents will be filtered (a real number between _0_ and _1_).
* `selection_lgb_model_dir` is the directory with the LGB model for clause selection,
* `weight_type` can be set to `1` to simply rate clauses as `positive` or `negative`.

Repositories to help with training models can be found at:

* https://github.com/zariuq/pyprove/tree/parentalguidance_frozen
* https://github.com/zariuq/enigmatic/tree/parentalguidance_frozen


## Requirements

The following Python packages are required (installable through `pip` or otherwise):

* `tensorflow` (1.5)
* `xopen`
* `asyncio`

