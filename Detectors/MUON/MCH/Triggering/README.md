<!-- doxy
\page refDetectorsMUONMCHTriggering Triggering
/doxy -->

# EventFinder.h(cxx)

Implementation of the MCH event finder algorithm.

## Input / Output

It takes as input the lists of MCH ROFs, MCH digits, MCH digit MC labels (if any) and MID ROFs. It returns the list of digits associated to an event, their MC labels (if any) and the list of ROFs corresponding to the events.

## Short description of the algorithm

The event finding is done in 4 steps:
1) Create an empty event for each MID ROF with a trigger window opened around the MID Interaction Record (IR). The size of the window is defined via EventFinderParam (see below).
2) Associate each MCH ROF to the first compatible event, if any. An event is compatible if the trigger window overlaps with the MCH RO window.
3) Merge overlapping events and remove empty ones. Two events overlap if the MCH RO window associated to one event overlaps with the trigger window of another.
4) Copy the digits (and MC labels) associated to each event in the ouput lists, merging them if the same pad is fired multiple times within the same event, and produce the output ROFs pointing to them. The time window of each ROF is set to contain only the MID IR(s) associated to the event.

# EventFinderParam.h(cxx)

Definition of the trigger window [min, max[, in BC unit, opened around to MID IR to gather the digits associated with this interaction.

The window is configurable via the command line or an INI or JSON configuration file (cf. [workflow documentation](../Workflow/README.md)).

## Example of workflow

`o2-mid-tracks-reader-workflow | o2-mch-reco-workflow --triggered`

This takes as input the root files with MCH digits and MID tracks and ROFs and gives as ouput a root file with the MCH tracks. The event finder device is activated in the reconstruction workflow with the option `--triggered`.

See the [workflow documentation](../Workflow/README.md) for more details about the event finder workflow.
