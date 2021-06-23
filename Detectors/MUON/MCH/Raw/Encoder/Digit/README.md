<!-- doxy
\page refDetectorsMUONMCHRawEncoderDigit Digit Conversion
/doxy -->

# MCH Digit to Raw Conversion

    o2-mch-digits-to-raw --input-file mchdigits.root

By default (at least currently) there's only one output file, named `mch.raw`
in the current directory.  The output directory can be changed using the
`--output-dir` option.

You can also get one file per link using the `--file-per-link` option, in which
 case the output files are named `mch_feeid####.raw`.

If you do not have a digit file at hand, for testing purposes you can do any
kind of simulation and digitization containing MCH, e.g.

    o2-sim -g fwmugen -m MCH -n 10
    o2-sim-digitizer-workflow --onlyDet MCH -b
