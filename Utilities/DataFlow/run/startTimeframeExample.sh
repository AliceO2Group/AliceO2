xterm -geometry 80x25+0+0 -hold -e FakeTimeframeGeneratorDevice --id FakeTimeframeGeneratorDevice --mq-config confFakeTimeframe.json --output-channel-name output &
xterm -geometry 80x25+0+0 -hold -e TimeframeWriterDevice --id TimeframeWriterDevice --mq-config confFakeTimeframe.json --input-channel-name input --max-timeframes 1 --output-file data.o2tf &
#xterm -geometry 80x25+0+0 -hold -e TimeframeReaderDevice --id TimeframeReaderDevice --mq-config confFakeTimeframe.json --input-file data.o2tf --output-channel-name output &
#xterm -geometry 80x25+0+0 -hold -e TimeframeValidatorDevice --id TimeframeValidatorDevice --mq-config confFakeTimeframe.json --input-channel-name input &
