This is a short documention for the DPL-DigitizerWorkflow example

# Status/Description of implementation

At present, the `digitizer-workflow` executable is a demonstrator of
how we intend to do initiate and handle the processing of hits, coming from detector simulation.

The digitizer-workflow currently demonstrates the transformation of hits into TPC digits using
realistic bunch crossing and collision sampling. We are also able to overlay hits from background and signal hit inputs.

The main components of the digitizer-workflow are

* The SimReader device:
  - reading/analysing the given hit files
  - performing the bunch crossing sampling and collision composition (stored in a collision context)
  - initiating digitization/processing by communicating the collision context to processing devices
  
* The TPC digitizier device:
  - producing digits in continuous time for a given sector
  - at present writes (or forwards) these digits in units of TPC drift times

The digitizer-workflow executable is already somewhat configurable, both in terms of
workflow/topology options as well as individual device options. Some help is available
via
```
digitizer-workflow --help
```
Other features are demonstrated in he following section.

# Feature example/Usage

Let's assume we have a background hit file `o2sim_bg.root` generated
by the O2 simulation with
```
o2sim -n 20 -g SOMEBACKGROUNDEVENTGENERATOR -m [detectors] -o o2sim_bg.root
```

Similar for a signal file `o2sim_sg.root`
```
o2sim -n 50 -g SOMESIGNALEVENTGENERATOR -m [detectors] -o o2sim_sg.root
```

1. **How can I digitize all sectors for the given background event?**
   ```
   digitizer-workflow -b --simFile o2sim_bg.root
   ```
   This will run as many TPC digitizer processors as there are logical CPU cores on your machine in parallel.
   (Note that depending on your available memory, this might cause problems as the digitization needs lots of memory; It might be safer to start with a small number of workers as indicated under point 3.).

2. **How can I only digitize sectors TPC sectors 1 + 2 for the given background event?**
   ```
   DPL_TPC_SECTORS=1,2 digitizer-workflow -b --simFile o2sim_bg.root
   ```

3. **How can I digitize sectors 1-8 using only 3 TPC digitizer devices?**
   ```
   DPL_TPC_SECTORS=1,2,3,4,5,6,7,8 DPL_TPC_LANES=3 digitizer-workflow -b --simFile o2sim_bg.root
   ```

4. **How can I digitize a total of 100 sampled collisions merging background and signal hits for TPC sector 1?**
   ```
   DPL_TPC_SECTORS=1 digitizer-workflow -b --simFile o2sim_bg.root --simFileS o2sim_sg.root -n 100
   ```

# Missing things/Improvements to come

At present the digitizer write individual digit files for each sector with names `tpc_digi_22_...`.
It is planned asap to make this more configurable and to outsource the writing to ROOT files in a different device.

Configuration of the workflow via environment variables is going to be substituted via a proper mechanism once this
is implemented by DPL.

Digitizers for other detectors shall be added.

The polay distribution should be commicated via CDB or some init mechanism.
