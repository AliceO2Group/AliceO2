<!-- doxy
\page refEventVisualisationWorkflow EventVisualisation Workflow
/doxy -->

# Event Visualisation Workflow

This module contains DPL worflow, which can be used to produce data (in JSON format) to be displayed be `o2-eve`. Workflow may be used:
* on EPN to produce data which are then copied to online visualisation machine in P2 (see process description \subpage refEventVisualisationScripts)
* locally to make visualisation based on simulated data


## Prepare data based on simulated data
before one starts steps described below, please verify that:
* for `o2-eve-display` step you should:
  * start within the folder where simulated data was produced
  * you need to store in the folder file ```ITSdictionary.bin``` and ```MFSdictionary.bin``` (available from O2-2288 jira)
* for `o2-eve` step you should:
  * have in the working folder:
    * `.o2eve_config` file (available from EventVisualisation/Scripts)
    * `o2sim_geometry.root` (copied from simulation folder)
    * `o2sim_grp.root` (copied from simulation folder)
  * have (somewhere) folder with json files (f.e `/home/ed/jsons`)
  * have (somewhere) folder with simplify geometry files (f.e `/home/ed/geom/O2`)



To visualise a simulated data one should:
* Run simulation
```shell
# for example:
enter dev
$O2_ROOT/prodtests/sim_challenge.sh -n 5 -s pbpb
```
* Run Workflow in folder with files produced by simulation
```shell
o2-global-track-cluster-reader --track-types TPC,ITS --cluster-types TPC,ITS | o2-eve-display --display-tracks TPC,ITS --display-clusters TPC,ITS
```
* Run `o2-eve` pointing a folder where produced `*.json` files were stored (see description # Event Visualisation View)
```shell
enter dev
o2-eve -j -d /home/ed/jsons -o
```


## o2-eve-display command line parameters:

| *parameter*| *default value*  | *description*  |  
|-----|---|---|
|jsons-folder     | jsons  |name of the host allowed to produce files   |
|eve-hostname     |   |name of the host allowed to produce files (empty means no limit)   |
|eve-dds-collection-index     |-1   |number of dpl collection allowed to produce files (-1 means no limit)   |  
|number-of_files     |300   |maximum number of json files in folder (newer one will replace oldest)   |  
|number-of_tracks     |-1   |maximum number of track stored in json file (-1 means no limit)   |  
|enable-mc     |false   |enable visualization of MC data   |  
|disable-mc     |false   |disable visualization of MC data   |  
|display-clusters     |ITS,TPC,TRD,TOF   |comma-separated list of clusters to display   |  
|display-tracks     |TPC,ITS,ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF   |comma-separated list of tracks to display   |  
|read-from-files     |false   |   |  
|disable-root-input     |false   |Disable root input overriding read-from-files   |  
|configKeyValues     |   |Semicolon separated key=value strings ..."   | 




