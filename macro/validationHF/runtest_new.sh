#!/bin/bash

#INPUTDIR="/data/Run3data/output" #K0* MC injected 

CASE=4
DOCONVERT=1
DORUN1=1
DORUN3=1
DOCOMPARE=1

if [ $CASE -eq 0 ]; then
  INPUTDIR="../twikiinput"
  ISMC=0
  LISTNAME="listprodhfrun3_ppK0starToyMC.txt"
  AOD3NAME=AO2D.root
  MASS=1.0
  STRING="AliESDs_ppK0starToyMC.root"
  TRIGGERSTRINGRUN2=""
  TRIGGERBITRUN3=-1
  NMAX=-1
  JSON=dpl-config_std.json
fi

if [ $CASE -eq 1 ]; then
  INPUTDIR="/mnt/temp/Run3data/data/LHC15o_246751/pass1"
  ISMC=0
  LISTNAME="listprodhfrun3_PbPb_data_LHC15o_246751.txt"
  AOD3NAME=AO2D.root
  MASS=1.8
  STRING="15000246751019.110/AliESDs.root"
  TRIGGERSTRINGRUN2="CV0L7-B-NOPF-CENT"
  TRIGGERBITRUN3=5 #FIXME
  NMAX=5
  JSON=dpl-config_std.json
fi

if [ $CASE -eq 2 ]; then
  INPUTDIR="/data/Run3data/alice_sim_2015_LHC15k1a3_246391/246391"
  ISMC=0
  LISTNAME="listprodhfrun3_mc_HIJING_PbPb_LHC15k1a3.txt"
  AOD3NAME=AO2D.root
  MASS=1.8
  STRING="00*/AliESDs.root"
  TRIGGERSTRINGRUN2=""
  TRIGGERBITRUN3=-1
  NMAX=1
  JSON=dpl-config_std.json
fi

if [ $CASE -eq 3 ]; then
  INPUTDIR="/data/Run3data/output"
  ISMC=0
  LISTNAME="listprodhfrun3_K0sMC.txt"
  AOD3NAME=AO2D.root
  MASS=1.0
  STRING="00*/AliESDs.root"
  TRIGGERSTRINGRUN2=""
  TRIGGERBITRUN3=-1
  NMAX=-1
  JSON=dpl-config_std.json
fi

if [ $CASE -eq 4 ]; then
  INPUTDIR="/data/Run3data/alice_sim_2018_LHC18a4a2_cent/282099"
  ISMC=0
  LISTNAME="listprodhfrun3_mc_PbPb_D2H_LHC18a4a2_cent.txt"
  AOD3NAME=AO2D.root
  MASS=1.8
  STRING="001/AliESDs.root"
  TRIGGERSTRINGRUN2=""
  TRIGGERBITRUN3=-1
  JSON=dpl-config_std.json
  NMAX=-1
fi

#INPUTDIR="/data/Run3data/alice_sim_2018_LHC18a4a2_cent/282099" #D2H MC sample
#INPUTDIR="/data/Run3data/alice_sim_2015_LHC15k1a3_246391/246391" #HIJING MC PbPb


rm *.root
rm *.txt
if [ $DOCONVERT -eq 1 ]; then
  rm $LISTNAME
  ls $INPUTDIR/$STRING >> $LISTNAME
  echo $LISTNAME
  root -q -l "convertAO2D.C(\"$LISTNAME\", $ISMC, $NMAX)"  
  mv AO2D.root $AOD3NAME
fi

if [ $DORUN1 -eq 1 ]; then
  rm Vertices2prong-ITS1_*.root 
  fileouttxt="outputlist.txt"
  rm $fileouttxt
  
  index=0
  while read F  ; do
    fileout="Vertices2prong-ITS1_$index.root"
    rm "$fileout"
    echo $fileout >> "$fileouttxt"
    echo "$F"
    echo "$fileout" 
    root -q -l "ComputeVerticesRun1_Opt.C(\"$F\",\"$fileout\",\"$JSON\")" 
    index=$((index+1))
    echo $index
  done <"$LISTNAME"
  rm "Vertices2prong-ITS1.root" 
  hadd Vertices2prong-ITS1.root @"$fileouttxt"

  
fi

if [ $DORUN3 -eq 1 ]; then
  rm AnalysisResults.root
  o2-analysis-hftrackindexskimscreator --shm-segment-size 16000000000 --configuration json://$PWD/dpl-config_std.json --aod-file $AOD3NAME | o2-analysis-hfcandidatecreator2prong --shm-segment-size 16000000000 --configuration json://$PWD/dpl-config_std.json --aod-file $AOD3NAME | o2-analysis-taskdzero --shm-segment-size 16000000000 --configuration json://$PWD/dpl-config_std.json --aod-file $AOD3NAME -b
  #o2-analysis-vertexing-hf --aod-file $AOD3NAME  -b --triggerindex=$TRIGGERBITRUN3
fi 

if [ $DOCOMPARE -eq 1 ]; then
  root -q -l "CompareNew.C(\"AnalysisResults.root\",\"Vertices2prong-ITS1.root\", $MASS)"
fi 

