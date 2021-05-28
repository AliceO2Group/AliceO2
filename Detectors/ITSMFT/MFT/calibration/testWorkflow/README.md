
# MFT calibration workflows

This will read data from raw data file, produces clusters and write a noise map to a local CCDB running on port 8080:

```shell
o2-raw-file-reader-workflow -b --delay ${DELAY_S} --nocheck-missing-stop --nocheck-starts-with-tf --nocheck-packet-increment --nocheck-hbf-jump --nocheck-hbf-per-tf --detect-tf0 --configKeyValues "HBFUtils.nHBFPerTF=${HBF_PER_TF}" --input-conf ${CFGFILE} | \
o2-itsmft-stf-decoder-workflow -b --nthreads ${N_THREAD} --runmft --decoder-verbosity ${DECODER_VERBOSITY} | \
o2-calibration-mft-calib-workflow --path "/MFT/test_CCDB/" --meta "Description=MFT;Author=Maurice Coquet;Uploader=Maurice Coquet" -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:8080" -b
```
The same as before but uses digits instead of clusters:
```shell
o2-raw-file-reader-workflow -b --delay ${DELAY_S} --nocheck-missing-stop --nocheck-starts-with-tf --nocheck-packet-increment --nocheck-hbf-jump --nocheck-hbf-per-tf --detect-tf0 --configKeyValues "HBFUtils.nHBFPerTF=${HBF_PER_TF}" --input-conf ${CFGFILE} | \
o2-itsmft-stf-decoder-workflow -b --nthreads ${N_THREAD} --runmft --digits --no-clusters --no-cluster-patterns --decoder-verbosity ${DECODER_VERBOSITY} | \
o2-calibration-mft-calib-workflow --useDigits --path "/MFT/test_CCDB/" --meta "Description=MFT;Author=Maurice Coquet;Uploader=Maurice Coquet" -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:8080" -b
```

Additional options to the mft-calib DPL :
```
  --path "/path/in/CCDB/" : defines path to write to in CCDB (default : "/MFT/Calib/NoiseMap")
  --meta "Description=...;Author=...;Uploader=..." : add meta data to the input in the CCDB
  --tstart <start timestamp> : defines the start of validity timestamp of the file written in the CCDB (default is -1 : current timestamp)
  --tend <end timestamp> : defines the start of validity timestamp of the file written in the CCDB (defult is -1 : one year from the current timestamp)
  --prob-threshold <proba> : defined probability threshold for noisy pixels (default : 3e-6)
```
