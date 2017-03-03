#### Prototype Devices for the transport between FLPs and EPNs
--------------------------------------------------------------

#### General

The devices implement following topology: 

In default mode:

![FLP2EPN topology](../../docs/images/flp2epn-distributed.png?raw=true "FLP2EPN topology")

In test mode:

![FLP2EPN topology test mode](../../docs/images/flp2epn-distributed-test-mode.png?raw=true "FLP2EPN topology in test mode")

- **flpSyncSampler** publishes timeframe IDs at configurable rate (only for the *test mode*).
- **flpSenders** generate dummy data of configurable size and distribute it to the available epnReceivers.
- **epnReceivers** collect all sub-timeframes (according to number of FLPs), merge them and send further.
- flpSenders choose which epnReceiver to send a given sub-timeframe to based on its ID (`timeframeId % NumEPNs`), ensuring that sub-timeframes with the same ID arrive at the same epnReceiver (without need for additional synchronization).
- Upon collecting sub-timeframes from all flpSenders, epnReceivers send confirmation to the sampler with the timeframe ID to measure roundtrip time (only for the *test mode*).
- epnReceivers can also measure intervals between receiving from the same FLP (used to see the effect of traffic shaping).
- The devices can run in *test mode* (as described above) and *default mode* where flpSenders receive data instead of generating it (as used by the Alice HLT devices).
- Optional deployment and execution via DDS.

#### Device configuration

The devices are configured via command line options and their connection parameters via JSON file. Most of the command line options have default values. These default values are for running in the *default mode*. Running in *test mode* requires a few modifications. Refer to `startFLP2EPN-distributed.sh.in` and `flp2epn-prototype.json` for example configuration.

These are the required device options:

**flpSyncSampler** (only for the *test mode*)

 - `--id arg`               Device ID
 - `--mq-config arg`        JSON configuration file

**flpSender**

 - `--id arg`               Device ID
 - `--mq-config arg`        JSON configuration file
 - `--num-epns arg`         Number of EPNs
 - only for test mode: `--test-mode arg (=0)`               "1" to run in test mode

**epnReceiver**

 - `--id arg`               Device ID
 - `--mq-config arg`        JSON configuration file
 - `--num-flps arg`         Number of FLPs
 - only for test mode: `--test-mode arg (=0)`               "1" to run in test mode

To list *all* available device options, run the executable with `--help`.

When running with DDS, configuration of addresses is also not required, because these are configured dynamically. Refer to `flp2epn-prototype-dds.json` and the DDS configuration files for an example.
