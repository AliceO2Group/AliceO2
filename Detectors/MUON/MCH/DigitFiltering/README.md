
<!-- doxy
\page refDetectorsMUONMCHDigitFiltering Digit filtering
/doxy -->

# MCH Digit Filtering

```shell
o2-mch-digits-filtering-workflow
```

Filter out some digits.

Inputs :
- list of all digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) in the current time frame, with the (default) data description `DIGITS` (can be changed with `--input-digits-data-description` option)
- the list of ROF records ([ROFRecord](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the digits associated to each interaction, with the (default) data description `DIGITROFS` (can be changed with `--input-digit-rofs-data-description` option)

Outputs :
- list of digits that pass the filtering criteria (for the moment ADC>0), with the (default) data description `F-DIGITS`  (can be changed with `--output-digits-data-description` option)
- list of ROF records corresponding to the digits above, with a (default) data description of `F-DIGITROFS` (can be changed with `--output-digit-rofs-data-description` option)

The exact behavior of the filtering is governed by the [MCHDigitFilterParam](/Detectors/MUON/MCH/DigitFiltering/include/MCHDigitFiltering/DigitFilterParam.h) configurable param, where you can select the minimum ADC value to consider, and whether to select signal (i.e. killing as much background as possible, possibly killing some signal as well) and/or to reject background (while not killing signal).
