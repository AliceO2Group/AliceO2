#### Prototype Devices for the transport between FLPs and EPNs
--------------------------------------------------------------

#### General

The devices implement following topology:

![FLP2EPN topology](../../docs/images/flp2epn-distr-rtt.png?raw=true "FLP2EPN topology")

- **flpSyncSampler** publishes timeframe IDs at configurable rate (only for the *test mode*).
- **flpSenders** generate dummy data of configurable size and distribute it to the available epnReceivers.
- **epnReceivers** collect all sub-timeframes (according to number of FLPs), merge them and send further.
- flpSenders choose which epnReceiver to send a given sub-timeframe to based on its ID (`timeframeId % NumEPNs`), ensuring that sub-timeframes with the same ID arrive at the same epnReceiver (without need for additional synchronization).
- Upon collecting sub-timeframes from all flpSenders, epnReceivers send confirmation to the sampler with the timeframe ID to measure roundtrip time.
- epnReceivers can also measure intervals between receiving from the same FLP (used to see the effect of traffic shaping).
- The devices can run in *test mode* (as described above) and *default mode* where flpSenders receive data instead of generating it (as used by the Alice HLT devices).
- Optional deployment and execution via DDS.

#### Device configuration

The devices are configured via command line options. Most of the command line options have default values. These default values are for running in the *default mode*. Running in *test mode* requires a few modifications.

These are the required device options:

**flpSyncSampler** (only for the *test mode*)

 - `--id arg`               Device ID
 - `--data-out-address arg` Data output address, e.g.: "tcp://localhost:5555"
 - `--ack-in-address arg`   Acknowledgement Input address, e.g.: "tcp://localhost:5556"

**flpSender**

 - `--id arg`               Device ID
 - `--num-epns arg`         Number of EPNs
 - `--data-in-address arg`  Data input address, e.g.:  "tcp://localhost:5555"
 - `--data-out-address arg` Data output address, e.g.: "tcp://localhost:5555"
 - `--hb-in-address arg`    Heartbeat input address, e.g.:  "tcp://localhost:5555"

Default overrides for *test mode*:

 - `--test-mode arg (=0)`               "1" to run in test mode
 - `--data-in-socket-type arg (=pull)`  "sub" for test mode
 - `--data-in-method arg (=bind)`       "connect" for test mode
 - `--data-in-address arg`  In test mode, the signal from flpSyncSampler sends to this address (data is generated).

**epnReceiver**

 - `--id arg`               Device ID
 - `--num-flps arg`         Number of FLPs
 - `--data-in-address arg`  Data input address, e.g.: "tcp://localhost:5555"
 - `--data-out-address arg` Output address, e.g.: "tcp://localhost:5555"
 - `--hb-out-address arg`   Heartbeat output address, e.g.: "tcp://localhost:5555"

Default overrides for *test mode*:

 - `--data-out-socket-type arg (=push)` "pub" for test mode (pub will just discard the data because no receiver in test mode).
 - `--ack-out-address arg`              For test mode provide address of the flpSyncSampler.

To list *all* available device options, run the executable with `--help`.

When running with DDS, configuration of addresses is also not required, because these are configured dynamically.

Example for the *test mode* can be found in `run/startFLP2EPN-distributed.sh.in` for manual run, or `runO2Prototype/flp_epn_topology.xml` for DDS run. For *default mode* there is an example DDS topology in `../topologies/o2prototype_topology.xml`.

