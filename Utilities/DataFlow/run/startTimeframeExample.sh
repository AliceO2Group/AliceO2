xterm -geometry 80x25+0+0 -hold -e FakeTimeframeGeneratorDevice --id FakeTimeframeGeneratorDevice --mq-config confFakeTimeframe.json --output-channel-name output &
xterm -geometry 80x25+0+0 -hold -e o2-timeframe-writer-device --id TimeframeWriterDevice --mq-config confFakeTimeframe.json --input-channel-name input --max-timeframes 1 --output-file data.o2tf &
#xterm -geometry 80x25+0+0 -hold -e o2-timeframe-reader-device --id TimeframeReaderDevice --mq-config confFakeTimeframe.json --input-file data.o2tf --output-channel-name output &
#xterm -geometry 80x25+0+0 -hold -e o2-timeframe-validator-device --id TimeframeValidatorDevice --mq-config confFakeTimeframe.json --input-channel-name input &
