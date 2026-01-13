<!-- doxy
\page refrunSimExamplesExternalToHybrid Example ExternalToHybrid
/doxy -->

This example demonstrates how to bypass the Hyperloop limitations when using external generators by switching the configuration to hybrid mode, using the new `GeneratorHybrid.switchExtToHybrid` parameter (set to false by default).

This solution works only with updated O2sim versions containing the `switchExtToHybrid` option.

# Configuration Files

Two example configuration files are provided, each pointing to different hybrid JSON files:

- **GeneratorHyperloopHybridCocktail.ini** → Creates a cocktail mixing two Pythia8 based generators and a boxgen instance
- **GeneratorHyperloopHybrid.ini** → Defines sequential generation of boxgen and EPOS4 events called with an external generator

# Script Description

## rundpl.sh

This script demonstrates event generation using the DPL framework, launching it with the external generator in hybrid mode.

### Available Flags

- **-i, --ini CONFIG** → Specifies the configuration ini file (default: `GeneratorHyperloopHybridCocktail.ini`)
- **-n, --nevents EVENTS** → Sets the number of events to generate (default: 5)
- **-h, --help** → Prints usage instructions and o2-sim-dpl-eventgen help
- **--** → Passes remaining command line arguments to o2-sim-dpl-eventgen

### Usage Examples

Run with default settings (5 events using cocktail configuration):
```bash
./rundpl.sh
```

Generate 10 events using the sequential configuration:
```bash
./rundpl.sh -n 10 -i ${O2_ROOT}/examples/ExternalToHybrid/GeneratorHyperloopHybrid.ini
```

# Requirements

- O2sim version with `switchExtToHybrid` support
- O2_ROOT and O2DPG_MC_CONFIG_ROOT environment variable must be loaded (possibly via O2sim directly)
- Appropriate external generator configurations (e.g., EPOS4) must be available