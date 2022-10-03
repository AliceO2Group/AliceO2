#!/usr/bin/env python3
#
# A script producing a consistent full-system-test workflow.
# Python used for convenient json handling but this is completely up to the workflow writer.
#
# WARNING: THIS SCRIPT IS CURRENTLY OUTDATED AND SHOULD BE MADE CONSISTENT WITH full_system_test.sh BEFORE USING IT AGAIN
#
#
from os import environ
import json

workflow={}
workflow['stages'] = []

taskcounter=0
def Task(name='', needs=[], tf=-1, cmd='a', cwd='./', lab=[]):
    global taskcounter
    taskcounter = taskcounter + 1
    t = { 'name': name, 'cmd': cmd, 'needs': needs, 'resources': { 'cpu': -1 , 'mem': -1 }, 'timeframe' : tf, 'labels' : lab, 'cwd' : cwd }
    workflow['stages'].append(t)
    return t

# ---- qed transport task -------
QED=Task(name='qedsim', lab=["QED", "SIM"], cwd='qed')
QED['cmd']='o2-sim --seed $O2SIMSEED -j $NJOBS -n$NEventsQED -m PIPE ITS MFT FT0 FV0 FDD -g extgen --configKeyValues \"GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/QEDLoader.C;QEDGenParam.yMin=-7;QEDGenParam.yMax=7;QEDGenParam.ptMin=0.001;QEDGenParam.ptMax=1.;Diamond.width[2]=6.\"'

QED2HAD=Task(name='qed2had', lab=["QED", "SIM"], cwd='qed', needs=["qedsim"])
QED2HAD['cmd']='PbPbXSec="8." ; awk "BEGIN {printf \\"%.2f\\",`grep xSectionQED qedgenparam.ini | cut -d\'=\' -f 2`/$PbPbXSec}" > qed2had.log'
#echo "Obtained ratio of QED to hadronic x-sections = $QED2HAD" >> qedsim.log

# --- signal sim and digitization ----
SIM=Task(name="sim", lab=["SIM"])
SIM['cmd']='o2-sim --seed $O2SIMSEED -n $NEvents --skipModules ZDC --configKeyValues "Diamond.width[2]=6." -g pythia8hi -j $NJOBS'

DIGI1=Task(name="digi", lab=["DIGI"], needs=["sim", "qed2had"])
DIGI1['cmd']='QED2HAD=`cat qed/qed2had.log`; echo ${QED2HAD}; o2-sim-digitizer-workflow -n $NEvents --simPrefixQED qed/o2sim --qed-x-section-ratio ${QED2HAD} ${NOMCLABELS} --firstOrbit 0 --firstBC 0 --skipDet TRD --tpc-lanes $((NJOBS < 36 ? NJOBS : 36)) --shm-segment-size $SHMSIZE ${GLOBALDPLOPT}'

# the dependency on digi is because of collisioncontext
DIGI2=Task(name='digiTRD', lab=["DIGI"], needs=["sim", "digi"])
DIGI2['cmd']='o2-sim-digitizer-workflow -n $NEvents ${NOMCLABELS} --firstOrbit 0 --firstBC 0 --onlyDet TRD --shm-segment-size $SHMSIZE ${GLOBALDPLOPT} --incontext collisioncontext.root --configKeyValues "TRDSimParams.digithreads=${NJOBS}"'

allrawtasknames=[]
def RAWTask(name, command):
    allrawtasknames.append(name)
    return Task(name=name, cmd=command, lab=["RAW"], needs=["digi"])

ITSRAW=RAWTask('itsraw', 'o2-its-digi2raw --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/ITS')
MFTRAW=RAWTask('mftraw', 'o2-mft-digi2raw --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/MFT')
FT0RAW=RAWTask('ft0raw', 'o2-ft0-digi2raw --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/FT0')
FV0RAW=RAWTask('fv0raw', 'o2-fv0-digi2raw --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/FV0')
FDDRAW=RAWTask('fddraw', 'o2-fdd-digit2raw --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/FDD')
TPCRAW=RAWTask('tpcraw', 'o2-tpc-digits-to-rawzs  --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -i tpcdigits.root -o raw/TPC')
TOFRAW=RAWTask('tofraw', 'o2-tof-reco-workflow ${GLOBALDPLOPT} --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" --output-type raw --tof-raw-outdir raw/TOF')
TOFRAW=RAWTask('midraw', 'o2-mid-digits-to-raw-workflow ${GLOBALDPLOPT} --mid-raw-outdir raw/MID --mid-raw-perlink  --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\"')
EMCRAW=RAWTask('emcraw', 'o2-emcal-rawcreator --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/EMC')
PHSRAW=RAWTask('phsraw', 'o2-phos-digi2raw  --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/PHS')
CPVRAW=RAWTask('cpvraw', 'o2-cpv-digi2raw  --file-for link --configKeyValues \"HBFUtils.nHBFPerTF=128;HBFUtils.orbitFirst=0\" -o raw/CPV')

# make configuration -> this depends on all previous raws
Task('rawAllConfig', cmd='cat raw/*/*.cfg > rawAll.cfg', needs=allrawtasknames, lab=["RAW"])

# now emit all possible versions of the dpl workflow (WITHGPU, NOGPU, AYNC)
RECO_ENV={ "WITHGPU": { "CREATECTFDICT":"0",
                        "GPUTYPE":"CUDA",
                        "GPUMEMSIZE":"6000000000",
                        "HOSTMEMSIZE":"1000000000",
                        "SYNCMODE":"1",
                        "CTFINPUT":"0",
                        "SAVECTF":"0" },
      "ASYNC": {
          "CREATECTFDICT":"0",
          "GPUTYPE":"CPU",
          "SYNCMODE":"0",
          "HOSTMEMSIZE":"$TPCTRACKERSCRATCHMEMORY",
          "CTFINPUT":"1",
          "SAVECTF":"0"
      },
      "NOGPU": {
          "CREATECTFDICT":"1",
          "GPUTYPE":"CPU",
          "SYNCMODE":"0",
          "HOSTMEMSIZE":"$TPCTRACKERSCRATCHMEMORY",
          "CTFINPUT":"0",
          "SAVECTF":"1"
      }
}

for stage in [ "NOGPU", "WITHGPU", "ASYNC" ]:
    t=Task(name='reco_' + stage, needs=['rawAllConfig'], lab=["RECO"])
    t['env']=RECO_ENV[stage]
    precommand=""
    if stage=="NOGPU":
        precommand="rm -f ctf_dictionary.root;"
    t['cmd']=precommand + "$O2_ROOT/prodtests/full-system-test/dpl-workflow.sh"

def trimString(cmd):
  return ' '.join(cmd.split())

# insert taskwrapper stuff
for s in workflow['stages']:
  s['cmd']='. ${O2_ROOT}/share/scripts/jobutils.sh; taskwrapper ' + s['name']+'.log \'' + s['cmd'] + '\''

# remove whitespaces etc
for s in workflow['stages']:
  s['cmd']=trimString(s['cmd'])

# write workflow to json
workflowfile='workflow.json'
with open(workflowfile, 'w') as outfile:
    json.dump(workflow, outfile, indent=2)

exit (0)
