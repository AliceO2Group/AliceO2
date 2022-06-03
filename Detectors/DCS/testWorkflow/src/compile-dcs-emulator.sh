ARCH=`aliBuild architecture`
WORK_DIR="$ALICE_WORK_DIR"
source ${WORK_DIR}/${ARCH}/cppzmq/latest/etc/profile.d/init.sh
source ${WORK_DIR}/${ARCH}/boost/latest/etc/profile.d/init.sh
g++ dcssend.cpp -o dcssend -I$BOOST_ROOT/include -I$CPPZMQ_ROOT/include -I$ZEROMQ_ROOT/include -L $BOOST_ROOT/lib -l boost_program_options -L$ZEROMQ_ROOT/lib -l zmq
g++ dcsclient.cpp -o dcsclient -I$BOOST_ROOT/include -I$CPPZMQ_ROOT/include -I$ZEROMQ_ROOT/include -L $BOOST_ROOT/lib -l boost_program_options -L$ZEROMQ_ROOT/lib -l zmq

