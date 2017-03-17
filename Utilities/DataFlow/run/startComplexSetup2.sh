# simple start script to launch a more complex setup
# with 2 data publishers (inside subframebuilder) + 2 attached flpSenders + 4 EPNS
# Prerequisites:
#  - expects the configuration file to be in the working directory
#  - O2 bin and lib set n the shell environment

# it would be nice having a script that generates the configuration file for N FLP and M EPNS

# Start one HBSampler device
xterm -geometry 80x25+0+0 -hold -e heartbeatSampler --id heartbeatSampler --mq-config confComplexSetup2.json --out-chan-name output &

# Data publishers
xterm -geometry 80x25+500+0 -hold -e DataPublisherDevice --id DataPublisherDeviceTPC --mq-config confComplexSetup2.json --in-chan-name input --out-chan-name output  --data-description TPCCLUSTER &
xterm -geometry 80x25+500+400 -hold -e DataPublisherDevice --id DataPublisherDeviceITS --mq-config confComplexSetup2.json --in-chan-name input --out-chan-name output --data-description ITSRAW  &


# this is the subtimeframe publisher for TPC
xterm -geometry 80x25+1000+0 -hold -e SubframeBuilderDevice --id subframeBuilderTPC --mq-config confComplexSetup2.json --detector TPC &

# this is the subtimeframe publisher for ITS
xterm -geometry 80x25+1000+400 -hold -e SubframeBuilderDevice --id subframeBuilderITS --mq-config confComplexSetup2.json --detector ITS &

# this is the flp for TPC
xterm -geometry 80x25+1500+0 -hold -e FLPSenderDevice --id flpSenderTPC --mq-config confComplexSetup2.json --in-chan-name input --out-chan-name output --num-epns 4 --flp-index 0 &

# this is the flp for ITS
xterm -geometry 80x25+1500+400 -hold -e FLPSenderDevice --id flpSenderITS --mq-config confComplexSetup2.json --in-chan-name input --out-chan-name output --num-epns 4 --flp-index 1 &

# we have 4 epn and 2 flps
xterm -geometry 80x25+2000+0 -hold -e EPNReceiverDevice --id epnReceiver1 --mq-config confComplexSetup2.json  --buffer-timeout 10000 --in-chan-name input --out-chan-name output --num-flps 2 &
xterm -geometry 80x25+2000+400 -hold -e EPNReceiverDevice --id epnReceiver2 --mq-config confComplexSetup2.json  --buffer-timeout 10000 --in-chan-name input --out-chan-name output --num-flps 2 &
xterm -geometry 80x25+2000+800 -hold -e EPNReceiverDevice --id epnReceiver3 --mq-config confComplexSetup2.json  --buffer-timeout 10000 --in-chan-name input --out-chan-name output --num-flps 2 &
xterm -geometry 80x25+2000+1200 -hold -e EPNReceiverDevice --id epnReceiver4 --mq-config confComplexSetup2.json  --buffer-timeout 10000 --in-chan-name input --out-chan-name output --num-flps 2 &

# consumer and validator of the full EPN time frame
xterm -geometry 80x25+2000+500 -hold -e TimeframeValidatorDevice --id timeframeValidator --mq-config confComplexSetup2.json --input-channel-name input &
