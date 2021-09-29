<!-- doxy
\page refEventVisualisationScripts EventVisualisation Scripts
/doxy -->

# Event Visualisation Scripts

## Configuration of the online event display in P2

There are 2 processes to be run on the visualisation machine in P2. To simplify deployment there are `bash` functions declared in `.bashrc` functions of the `ed` user. 

* Process which synchronises (every 2 seconds) local folder  `/home/ed/jsons` with folder on EPN which is run in terminal by calling `query-alice`. That process should be run from one of the terminals. Because files in folder on EPN are produced in FIFO scheme (so creating new ones results in deletion the same number of the old ones) files in local folders also are created and deleted.
```shell
query-alice
```
* Process which starts `o2-eve` application in mode when contents of the `/home/ed/jsons` folder is observed which is run in terminal by calling `o2eve`. Before calling `o2eve` user must enter O2 environment - it can be done by calling `enter` function declared in `.bashrc`
```shell
enter
o2eve
```
## navigation in o2-eve events using << < > >> buttons 
Event display shows defined pool of data (by default there are 300+2) which are present in observed folder. The first and last data are duplicated (so 1 and 2 show the same and 301 and 302 show the same) because:
* position 1 means 'show the latest available data'. When observed latest data disappear (because new data has arrived and old one deleted) the new latest data is displayed
* positions 2-301: as long as data exists in the folder o2-eve displays the same data (but their positions decreases as new data is arriving and old data is deleted). Finally, the position become 1.
* position 302 means 'show the newest available data'. If we are on position 302 as soon as new data file is presented in the observed folder that data is immediately displayed by the o2-eve. The position remains the same, but the data displayed changes.
