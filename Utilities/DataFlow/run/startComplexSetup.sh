# simple start script to launch a more complex setup
# with 2 data publishers (inside subframebuilder) + 2 attached flpSenders
# Prerequisites:
#  - expects the configuration file to be in the working directory
#  - O2 bin and lib set n the shell environment


# we have just one epn and 2 flps 
xterm -geometry 80x25+0+0 -hold -e epnReceiver --id epnReceiver --mq-config confComplexSetup.json --in-chan-name input --out-chan-name output --num-flps 2 &

# this is the flp for TPC
xterm -geometry 80x25+500+0 -hold -e flpSender --id flpSenderTPC --mq-config confComplexSetup.json --in-chan-name input --out-chan-name output --num-epns 1 --flp-index 0 &

# this is the flp for ITS
xterm -geometry 80x25+800+0 -hold -e flpSender --id flpSenderITS --mq-config confComplexSetup.json --in-chan-name input --out-chan-name output --num-epns 1 --flp-index 1 &

# this is the subtimeframe publisher for TPC
xterm -geometry 80x25+0+500 -hold -e SubframeBuilderDevice --id subframeBuilderTPC --mq-config confComplexSetup.json --self-triggered --detector TPC &

# this is the subtimeframe publisher for ITS
xterm -geometry 80x25+500+500 -hold -e SubframeBuilderDevice --id subframeBuilderITS --mq-config confComplexSetup.json --self-triggered --detector ITS &

# consumer and validator of the full EPN time frame
xterm -geometry 80x25+800+500 -hold -e TimeframeValidatorDevice --id timeframeValidator --mq-config confComplexSetup.json --input-channel-name input &
