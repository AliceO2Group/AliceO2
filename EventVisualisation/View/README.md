<!-- doxy
\page refEventVisualisationView EventVisualisation View
/doxy -->

# Event Visualisation View

To run `o2-eve` you need to have a computer with linux or mac with properly build O2. There are no special hardware requirements, but for performance it is better to have discrete graphic card.

## Prerequisites
for `o2-eve` to run you should:
* have in the working folder:
    * `.o2eve_config` file (available from EventVisualisation/Scripts)
    * `o2sim_geometry.root` (copied from simulation folder)
    * `o2sim_grp.root` (copied from simulation folder)
* have (somewhere, specified by command line parameter) folder with json files (f.e `/home/ed/jsons`)
* have (somewhere, specified in `.o2eve_config` file) folder with simplify geometry files (f.e `/home/ed/geom/O2`)

## Running o2-eve
Here are sample commands:
```shell
alienv enter O2/latest-dev-o2
cd /home/ed     # working folder containing .o2eve_config, o2sim_geometry.root and o2sim_grp.root
o2-eve -o -d /home/ed/jsons
```

## o2-eve command line parameters:
| *parameter*|  *description*  |  *status*  |
|-----|---|---|
|i |displaying ITS tracks from `o2trac_its.root` and clusters from `o2clus_its.root`  |(under development)    |
|j |reading from *.json files (non online mode). require specification of json folder with `-d`  |   |
|o |online mode. require specification of json folder with `-d`  |   |
|r |randomly generated tracks   | (test option, switched off)  |
|t |displaying TPC tracks from `o2trac_its.root` and clusters from `o2clus_its.root`   |(under development)   |
|v |displaying data from sample ROOT file (vsd)   | (development test usage only)  |
|f file |location of AOD file containing data to be displayed   | (under development)  |
|d folder|location of json folder - a folder where json files from workflow data are located  |   |
