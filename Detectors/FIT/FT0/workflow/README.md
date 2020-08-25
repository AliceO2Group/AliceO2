<!-- doxy
\page refFITFT0workflow FIT/FT0 FLP DPL workflow
/doxy -->

# FLP DPL workflow, for reading raw data blocks from payload, and converting them into digits.

To run with source file:

o2-raw-file-reader-workflow -b --input-conf /home/fitdaq/work/data_raw_reader/run_raw_reader_v6.cfg|o2-ft0-flp-dpl-workflow -b

If you want to dump digits in reader DPL:

o2-raw-file-reader-workflow -b --input-conf /home/fitdaq/work/data_raw_reader/run_raw_reader_v6.cfg|o2-ft0-flp-dpl-workflow -b --dump-blocks-reader

If you need to check data at "proccessor" DPL:

o2-raw-file-reader-workflow -b --input-conf /home/fitdaq/work/data_raw_reader/run_raw_reader_v6.cfg|o2-ft0-flp-dpl-workflow -b --dump-blocks-process --use-process

Special TCM extended mode (only for special technical runs):

o2-raw-file-reader-workflow -b --input-conf /home/fitdaq/work/data_raw_reader/run_raw_reader_v6.cfg|o2-ft0-flp-dpl-workflow -b --tcm-extended-mode