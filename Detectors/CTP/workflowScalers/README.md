<!-- doxy
\page refDetectorsCTP CTP Workflow Scalers
/doxy -->

How to generate c++ proto files ?
On the computer where project
https://gitlab.cern.ch/aliceCTP3/ctp3-ipbus/-/tree/master
is installed, run

protoc -I$CTP3_ROOT/ccm/protos  --cpp_out=. $CTP3_ROOT/ccm/protos/ctpd.proto

The h and cc files are in current directory.
