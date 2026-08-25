<!-- doxy
\page refrunSimExamplesEventPool_Chaining Example EventPool_Chaining
/doxy -->

# Chaining of event pools

The `evtpool` generator goes through the **whole pool**, one file at a time: it starts
with the first one and, once its events are exhausted, moves on to the next. The files
are opened strictly one at a time and only when they are actually needed — a pool of a
thousand files costs exactly one open file handle, and a job that never gets past the
first file never touches the others. This is what is needed on hyperloop, where many
events have to be served from a pool made of many small files.

# Configuration

| key | default | meaning |
| --- | --- | --- |
| `GeneratorEventPool.eventPoolPath` | `""` | pool directory, a text file with a list of files, or a single `evtpool.root` |
| `GeneratorEventPool.randomize` | `true` | serve the events of each file in random order (a permutation, every event exactly once) |
| `GeneratorEventPool.roundRobin` | `false` | start over with the first file once the last one is exhausted |
| `GeneratorEventPool.rngseed` | `0` | seed used both for the order in which files are visited and for the event randomization |

The order in which the pool files are visited is shuffled per job, so that different jobs
of the same production do not all read the same files in the same order. Using a fixed
`rngseed` makes that order reproducible. No file is opened at initialisation time apart
from the first one — the shuffle only picks *names*.

Order in which events are served — it is fixed for a file at the moment that file is
opened, and every one of its events is used **exactly once** (different behaviour than the past):

* `randomize=false`: entry 0, 1, 2, … of the first file, then entry 0, 1, 2, … of the
  second file, and so on;
* `randomize=true` (the default): a random permutation of the entries of the first file,
  then a random permutation of the entries of the second one, and so on.

Once the last file has been used up the job **fails with a fatal error** unless
`roundRobin=true` is set. Making sure that the pool holds enough events for the
requested `-n` is the responsibility of the user:

```
[FATAL] GeneratorFromO2Kine: ran out of events after 6 event(s) from 3 input file(s)
        (9 were requested). Provide more input files/events or allow reusing them via roundRobin
```

With `roundRobin=true` the generator starts over from the first file; in randomized mode
a fresh permutation is drawn on every pass.

Example:

```bash
o2-sim -n 100 -g evtpool --configKeyValues "GeneratorEventPool.eventPoolPath=/path/to/pool"
```

The same works for AliEn pools:

```bash
o2-sim -n 100 -g evtpool \
       --configKeyValues "GeneratorEventPool.eventPoolPath=alien:///alice/cern.ch/user/.../evtpool_dir"
```

and inside the hybrid generator JSON configuration:

```json
{
  "name": "evtpool",
  "config": {
    "eventPoolPath": "/path/to/pool",
    "skipNonTrackable": true,
    "roundRobin": false,
    "randomize": false,
    "rngseed": 0,
    "randomphi": false
  }
}
```

## Use from the DPL event generator (hyperloop)

The very same configuration works with `o2-sim-dpl-eventgen`, which is the entry point
used on hyperloop:

```bash
o2-sim-dpl-eventgen -b --nEvents 1000 --generator evtpool --vertexMode kNoVertex \
    --configKeyValues "GeneratorEventPool.eventPoolPath=/path/to/pool" |\
  o2-sim-mctracks-to-aod -b |\
  o2-analysis-mctracks-to-aod-simple-task -b
```

`rundpl.sh` in this folder runs that pipeline for the pool created by `run.sh`.

Note that the seed given to the driver (`o2-sim --seed` / `o2-sim-dpl-eventgen --seed`)
governs both the order in which the pool files are visited and the event order inside them,
as long as `GeneratorEventPool.rngseed` is left at its default of 0. Setting `rngseed`
explicitly overrides the driver seed for the event pool.

Example of a workflow reading from an event pool:

```bash
${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py -eCM 900 -col pp -gen evtpool -tf 1 -ns 7 \
    -e TGeant4 -j 4 -interactionRate 50000 -run 300000 -seed 12345 \
    -confKey "GeneratorEventPool.eventPoolPath=/path/to/pool"
${O2DPG_ROOT}/MC/bin/o2dpg_workflow_runner.py -f workflow.json -tt aod
```

## Reading several plain kinematics files (`extkinO2`)

The same machinery is available for the ordinary `extkinO2` generator by passing a
comma-separated list of files, which are again read one after the other:

```bash
o2-sim -n 100 -g extkinO2 --extKinFile "kine1.root,kine2.root,kine3.root"
# or
o2-sim -n 100 -g extkinO2 --configKeyValues "GeneratorFromO2Kine.fileName=kine1.root,kine2.root"
```

# Provenance of the events

Every generated event stores the file it was read from and the entry inside that file in
its MC event header:

* `forwarding-generator_inputFile`
* `forwarding-generator_inputEventNumber` (entry within that file)

# Files description

- **run.sh** &rarr; creates a small event pool and reads it back
- **rundpl.sh** &rarr; the same through `o2-sim-dpl-eventgen` (the hyperloop path)
