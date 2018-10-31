# starts a simulation device setup
# just for testing purposes --> use o2sim_parallel for production
NSIMWORKERS=$1

killall -9 xterm
killall -9 O2PrimaryServerDeviceRunner
killall -9 O2SimDeviceRunner
killall -9 O2HitMergerRunner

topologyfile=${O2_ROOT}/share/config/o2simtopology.json

# we have one primary distributor 
xterm -geometry 80x25+0+0 -e "O2PrimaryServerDeviceRunner --control static --id primary-server --mq-config ${topologyfile} -n 20 -m PIPE TOF TRD TPC PHS EMC FIT MCH -g pythia8 -e TGeant3 | tee serverlog;bash" &

for i in `seq 1 ${NSIMWORKERS}`; do
  xterm -geometry 80x25+500+0 -e "O2SimDeviceRunner --control static --id worker${i} --config-key worker --mq-config ${topologyfile} --severity info  | tee simlog${i};bash" &
done


# the its digitizer
#xterm -geometry 80x25+1000+0 -hold -e "O2ITSDigitizerDeviceRunner --control static --id itsdigitizer --mq-config ${topologyfile} &


# one hit merger -> the time measures the walltime of the complete session
time O2HitMergerRunner --id hitmerger --control static --mq-config ${topologyfile} | tee mergelog 

