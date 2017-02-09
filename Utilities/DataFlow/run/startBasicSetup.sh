# simple start script to lauch a basic setup
# Prerequisites:
#  - expects the configuration file to be in the working directory
#  - O2 bin and lib set n the shell environment


xterm -geometry 80x25+0+0 -hold -e epnReceiver --id epnReceiver --mq-config confBasicSetup.json --in-chan-name input --out-chan-name output --num-flps 1 &

xterm -geometry 80x25+0+0 -hold -e flpSender --id flpSender --mq-config confBasicSetup.json --in-chan-name input --out-chan-name output --num-epns 1 &

xterm -geometry 80x25+0+0 -hold -e SubframeBuilderDevice --id subframeBuilder --mq-config confBasicSetup.json --self-triggered &
