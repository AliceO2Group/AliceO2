#!/bin/bash

# Make sure we can open sufficiently many files / allocate enough memory
ulimit -n 4096 && ulimit -l unlimited && ulimit -m unlimited && ulimit -l unlimited
if [ $? != 0 ]; then
  echo Error setting ulimits
  exit 1
fi

if [ -z "$NTIMEFRAMES" ];   then export NTIMEFRAMES=1; fi              # Number of time frames to process
if [ -z "$TFDELAY" ];       then export TFDELAY=100; fi                # Delay in seconds between publishing time frames
if [ -z "$NGPUS" ];         then export NGPUS=1; fi                    # Number of GPUs to use, data distributed round-robin
if [ -z "$GPUTYPE" ];       then export GPUTYPE=CPU; fi                # GPU Tracking backend to use, can be CPU / CUDA / HIP / OCL / OCL2
if [ -z "$SHMSIZE" ];       then export SHMSIZE=$(( 128 << 30 )); fi   # Size of shared memory for messages
if [ -z "$DDSHMSIZE" ];     then export DDSHMSIZE=$(( 32 << 10 )); fi  # Size of shared memory for DD Input
if [ -z "$GPUMEMSIZE" ];    then export GPUMEMSIZE=$(( 13 << 30 )); fi # Size of allocated GPU memory (if GPUTYPE != CPU)
if [ -z "$HOSTMEMSIZE" ];   then export HOSTMEMSIZE=0; fi              # Size of allocated host memory for GPU reconstruction (0 = default)
if [ -z "$CREATECTFDICT" ]; then export CREATECTFDICT=0; fi            # Create CTF dictionary
if [ -z "$SAVECTF" ];       then export SAVECTF=0; fi                  # Save the CTF to a ROOT file
if [ -z "$SYNCMODE" ];      then export SYNCMODE=0; fi                 # Run only reconstruction steps of the synchronous reconstruction
if [ -z "$NUMAID" ];        then export NUMAID=0; fi                   # SHM segment id to use for shipping data as well as set of GPUs to use (use 0 / 1 for 2 NUMA domains)
if [ -z "$NUMAGPUIDS" ];    then export NUMAGPUIDS=0; fi               # NUMAID-aware GPU id selection
if [ -z "$EXTINPUT" ];      then export EXTINPUT=0; fi                 # Receive input from raw FMQ channel instead of running o2-raw-file-reader
if [ -z "$CTFINPUT" ];      then export CTFINPUT=0; fi                 # Read input from CTF (incompatible to EXTINPUT=1)
if [ -z "$NHBPERTF" ];      then export NHBPERTF=128; fi               # Time frame length (in HBF)
if [ -z "$GLOBALDPLOPT" ];  then export GLOBALDPLOPT=; fi              # Global DPL workflow options appended at the end
if [ -z "$EPNPIPELINES" ];  then export EPNPIPELINES=0; fi             # Set default EPN pipeline multiplicities
if [ -z "$SEVERITY" ];      then export SEVERITY="info"; fi            # Log verbosity
if [ -z "$SHMTHROW" ];      then export SHMTHROW=1; fi                 # Throw exception when running out of SHM
if [ -z "$NORATELOG" ];     then export NORATELOG=1; fi                # Disable FairMQ Rate Logging

SEVERITY_TPC="info" # overrides severity for the tpc workflow
DISABLE_MC="--disable-mc"

if [ $EXTINPUT == 1 ] && [ $CTFINPUT == 1 ]; then
  echo EXTINPUT and CTFINPUT are incompatible
  exit 1
fi
if [ $SAVECTF == 1 ] && [ $CTFINPUT == 1 ]; then
  echo SAVECTF and CTFINPUT are incompatible
  exit 1
fi
if [ $SYNCMODE == 1 ] && [ $CTFINPUT == 1 ]; then
  echo SYNCMODE and CTFINPUT are incompatible
  exit 1
fi
