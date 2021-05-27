<!-- doxy
\page refDetectorsMUONMCHCTF MCH CTF encoding library
/doxy -->

# MCH CTF

This directory contains the classes to handle entropy encoding and decoding of
MCH digits (and their associated ROFRecord).

# Circular test

To idea of this test is to check that digits data stays unaltered when they go
through the encoding/decoding processing.

```
circular-test.sh [NROFPERTF] [NTF] [OCC] [SEED]
```

The [circular-test.sh](./test/circular-test.sh) script performs for a given set of
{ number of timeframes, number of rofs per timeframe, occupancy, seed }, a
variation of the operations described below.

## Generate random digits and write them (debug binary format)

```
o2-mch-digits-random-generator-workflow -b
  --nof-rofs-per-tf 3
  --max-nof-tfs 5
  --occupancy 0.01
  --seed 1
| o2-mch-digits-writer-workflow -b
  --binary-file-format 1
  --outfile digits_ref_rof_3_tf_5_occ_0.01_seed_1.data
```

Note the `--seed` option that _must_ be non-zero if you want to get
reproducible results when running several times.


## Read those digits and encode them in a CTF

```
o2-mch-digits-file-reader-workflow -b
  --infile digits_ref.data
| o2-mch-entropy-encoder-workflow -b
| o2-ctf-writer-workflow -b
  --onlyDet MCH --no-grp
```

## Read back the CTF, decode the digits, write them

```
o2-ctf-reader-workflow -b
  --ctf-input o2_ctf_run00000000_orbit0000000000_tf0000000000.root
| o2-mch-digits-writer-workflow -b
  --binary-file-format 1
  --outfile digits.data
```

## Dump the written digits in text form

```
o2-mch-digits-file-reader-workflow -b
  --infile digits_ref.data
| o2-mch-digits-writer-workflow -b
  --no-file
  --txt
  --print-digits
```

## Compare digits before after ctf encoding/decoding

```
ls -al digits.*data
shasum -a 256 digits*.data
```


