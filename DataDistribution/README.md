# Data Distribution


Data Distribution O2 devices (processes) implement SubTimeFrame building on FLPs, and TimeFrame aggregation on EPNs.

## Components

Three components of the data distribution chain are:

- `SubTimeFrameBuilderDevice` (FLP): The first device in the chain. It receives readout data and once all inputs for an STF are received, forwards the STF into DPL or directly to `SubTimeFrameSenderDevice`
- `SubTimeFrameSenderDevice` (FLP):  Receives STF data and related results of local processing on FLP and performs TimeFrame aggregating.
- `TimeFrameBuilderDevice` (EPN): Receives STFs from all `SubTimeFrameSenderDevice`, creates a full TimeFrame and forwards it to global processing.

## Running with the Readout CRU emulator

Make sure the Readout module is loaded in the environment (make sure the `readout.exe` executable exists).

Run the chain with the `bin/start_Emulator-3FLP-3EPN.sh` script. The script supports running up to 3 independent FLP chains (CRU emulator, SufBuilder, StfSender) and up to 3 EPN TfBuilders on a local machine.
For supported options pass `--help` to the start script.

O2Device channel configuration is in `bin/config/readout-emu-flp-epn-chain.json`.  If using CRU emulation mode of the `readout.exe` process, configuration of the emulator is read from `bin/config/readout_cfg/readout_emu.cfg`, sections `[consumer-fmq-wp5]`, `[equipment-emulator-1]`, and `[equipment-emulator-2]`. To enable testing with two equipments, set the `[equipment-emulator-2]::enabled`  option to `1`.
