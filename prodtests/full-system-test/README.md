<!-- doxy
\page refprodtestsfull-system-test Full system test configuration and scripts
/doxy -->

## Full system test configuration and scripts

The full system test workflow scripts consist of 3 shell scripts:
* `dpl-workflow.sh` : The main script that runs the dpl-workflow for the reconstruction.
   It can read the input either internally, or receive it externally by one of the others.
* `raw-reader.sh` : Runs the `o2-raw-file-reader` to read the raw files as external input to `dpl-workflow.sh`.
* `datadistribution.sh` : Run the `StfBuilder` to read time frame files as external input to `dpl-workflow.sh`.

One can either run the `dpl-workflow.sh` standalone (with `EXTINPUT=0`) or in parallel with one of the other scripts in separate shells (with `EXTINPUT=1`)

In addition, there is the shared `setenv.sh` script which sets default configuration options, and there is the additional benchmark script:
* `start_tmux.sh` : This starts the full test in the configuration for the EPN with 2 NUMA domains, 512 GB RAM, 8 GPUs.
   It will run tmux with 3 sessions, running twice the `dpl-workflow.sh` and once one of the external input scripts (selected via `dd` and `rr` command line option).
   * Please note that `start_tmux.sh` overrides several of the environment options (see below) with the defaults for the EPN.
     The only relevant options for `start_tmux.sh` should be `TFDELAY` and `GPUMEMSIZE`.
   * Note also that while `dpl-workflow.sh` is a generic flexible script that can be used for actual operation, `start_tmux.sh` is a benchmark script to demonstrate how the full workflow is supposed to run on the EPN.
     It is meant for standalone tests and not to really start the actual processing on the EPN.

The `dpl-workflow.sh` can run both the synchronous and the asynchronous workflow, selected via the `SYNCMODE` option (see below), but note the following constraints.
* By default, it will run the full chain (EPN + FLP parts) such that it can operate as a full standalone benchmark processing simulated raw data.
* In order to run only the EPN part (skipping the steps that will run on the FLP), an `EPNONLY` option will be added later.

All settings are configured via environment variables.
The default settings (if no env variable is exported) are defined in `setenv.sh` which is sourced by all other scripts.
(Please note that `start_tmux.sh` overrides a couple of options with EPN defaults).
The following options exist (some of the options are not used in all scripts, and might behave slightly differently as noted):
* `NTIMEFRAMES`: Number of time frames to process.
  * `dpl-workflow.sh` without `EXTINPUT`: Will replay the timeframe `NTIMEFRAMES` time and then exit.
  * `raw-reader.sh` : Will replay the timeframe `NTIMEFRAMES` time and `raw-reader.sh` will exit, the dpl-workflows will keep running.
  * Ignored by `datadistribution.sh`, it will always run in an endless loop.
* `TFDELAY`: Delay in seconds between publishing time frames (1 / rate).
* `NGPUS`: Number of GPUs to use, data distributed round-robin.
* `GPUTYPE`: GPU Tracking backend to use, can be CPU / CUDA / HIP / OCL / OCL2.
* `SHMSIZE`: Size of the global shared memory segment.
* `DDSHMSIZE`: Size of shared memory unmanaged region for DataDistribution Input.
* `GPUMEMSIZE`: Size of allocated GPU memory (if GPUTYPE != CPU)
* `HOSTMEMSIZE`: Size of allocated host memory for GPU reconstruction (0 = default).
  * For `GPUTYPE = CPU`: TPC Tracking scratch memory size. (Default 0 -> dynamic allocation.)
  * Otherwise : Size of page-locked host memory for GPU processing. (Defauls 0 -> 1 GB.)
* `SAVECTF`: Save the CTF to a root file.
* `CREATECTFDICT`: Create CTF dictionary.
  * 0: Read `ctf_dictionary.root` as input.
  * 1: Create `ctf_dictionary.root`. Note that this was already done automatically if the raw data was simulated with `full_system_test.sh`.
* `SYNCMODE`: Run only reconstruction steps of the synchronous reconstruction.
* `NUMAGPUIDS`: NUMAID-aware GPU id selection. Needed for the full EPN configuration with 8 GPUs, 2 NUMA domains, 4 GPUs per domain.
  In this configuration, 2 instances of `dpl-workflow.sh` must run in parallel.
  To be used in combination with `NUMAID` to select the id per workflow.
  `start_tmux.sh` will set up these variables automatically.
* `NUMAID`: SHM segment id to use for shipping data as well as set of GPUs to use (use `0` / `1` for 2 NUMA domains, 0 = GPUS `0` to `NGPUS - 1`, 1 = GPUS `NGPUS` to `2 * NGPUS - 1`)
* 0: Runs all reconstruction steps, of sync and of async reconstruction, using raw data input.
* 1: Runs only the steps of synchronous reconstruction, using raw data input.
* `EXTINPUT`: Receive input from raw FMQ channel instead of running o2-raw-file-reader.
  * 0: `dpl-workflow.sh` can run as standalone benchmark, and will read the input itself.
  * 1: To be used in combination with either `datadistribution.sh` or `raw-reader.sh` or with another DataDistribution instance.
* `NHBPERTF`: Time frame length (in HBF)
* `GLOBALDPLOPT`: Global DPL workflow options appended to o2-dpl-run.
* `EPNPIPELINES`: Set default EPN pipeline multiplicities.
  Normally the workflow will start 1 dpl device per processor.
  For some of the CPU parts, this is insufficient to keep step with the GPU processing rate, e.g. one ITS-TPC matcher on the CPU is slower than the TPC tracking on multiple GPUs.
  This option adds some multiplicies for CPU processes using DPL's pipeline feature.
  The settings were tuned for EPN processing with 8 GPUs.
  It is auto-selected by `start-tmux.sh`.
* `SEVERITY`: Log verbosity (e.g. info or error)
* `SHMTHROW`: Throw exception when running out of SHM memory.
  It is suggested to leave this enabled (default) on tests on the laptop to get an actual error when it runs out of memory.
  This is disabled in `start_tmux.sh`, to avoid breaking the processing while there is a chance that another process might free memory and we can continue.
* `NORATELOG`: Disable FairMQ Rate Logging.
