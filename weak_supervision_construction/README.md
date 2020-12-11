# Weak Supervision Construction

## Download CNN/DM dataset
```shell
bash download/download_cnndm.sh
```

## Download ConceptNet Knowledge Graph
```shell
bash download/download_concept_net.sh
```
It is a subset of ConceptNet extracted by removing all non-English terms.

## Construct Dataet
```shell
python main.py --split train (or dev/test)
```
here ```split``` means using which split of CNN/DM dataset for the construction.