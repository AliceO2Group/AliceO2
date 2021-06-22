# Event filter for diffractive events

Checks for diffractive events and fills table `DiffractionFilters`. The table is defined in EventFiltering/filterTables.h.

The diffractive event filter is contained in the following set of files:

### cutHolder.cxx, cutHolder.h
`cutHolder` is a buffer for the various cut parameters. Is used to create a configurable which values can be set by command line options.

### diffractionSelectors.h
Contains the actual filter logics for the various types of events. Currently implemented are:

- DGSelector: a filter for Double Gap (DG) events

### diffractionFilter.cxx
Contains the actual filter tasks. Currently implemented are:
- DGFilterRun2: filter task for DG events in Run 2 data
- DGFilterRun3: filter task for DG events in Run 3 data
