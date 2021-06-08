<!-- doxy
\page refHMPIDworkflow HMP workflow
/doxy -->

# DPL workflows for the HMPID  v.0.7

## HMPID DPL processors

* `hmpid-raw-to-digits` reads data from ReadOut Raw File and outputs a ROOT formatted file (Reconstruction flow)
* `hmpid-digits-to-raw` reads digits data from a ROOT formatted file and produce a ReadOut Raw File (Simulation flow)
* `hmpid-read-raw-file-stream` reads data from ReadOut Raw File and outputs a stream of RDH6 pages as HMP/RAWDATA stream
* `hmpid-raw-to-digits-stream` decodes the input raw pages stream (HMP/RAWDATA) and produces two streams avector of o2::hmpid::Digits (HMP/DIGITS) and a o2::InteractionRecord (HMP/INTRECORD)
* `hmpid-digits-to-raw-stream` codes input o2::hmpid::Digits vector (HMP/DIGITS) and (HMP/INTRECORD) stream into the binary HMPID raw file
* `hmpid-dump-digits-stream` dumps the input o2::hmpid::Digits vector (HMP/DIGITS) and (HMP/INTRECORD) stream to the stdout or in an ASCII file
* `hmpid-raw-to-pedestals` reads data from ReadOut Raw File and outputs results of Pedestals calculation in ReadOut format and CCDB (Calibration flow)

### Workflow example
Dumping the contents of a Raw file.

```
 o2-hmpid-read-raw-file-stream-workflow --raw-file test_full_flp1.raw -b | o2-hmpid-raw-to-digits-stream-workflow -b | o2-hmpid-dump-digits-stream-workflow --out-file /tmp/pippo.txt -b
```

This reads the `test_full_flp1.raw` file and after the decoding, produce an ASCII file in the /tmp/ folder that contains the Events/Digits dump.



### o2-hmpid-read-raw-file-stream-workflow
Reads data from ReadOut Raw File and outputs a stream of RDH6 pages as HMP/RAWDATA stream.

Display all options

```
o2-hmpid-read-raw-file-stream-workflow --help full
```

Data processor options: HMP-ReadRawFile:

```
  --raw-file arg                        Raw input file name
  --print                               verbose output
```

### o2-hmpid-raw-to-digits-stream-workflow
Decodes the input raw pages stream (HMP/RAWDATA) and produces a vector of o2::hmpid::Digits (HMP/DIGITS) and a o2::InteractionRecord (HMP/INTRECORD).

Display all options

```
o2-hmpid-raw-to-digits-stream-workflow --help full
```

Data processor options: HMP-DataDecoder:

```
  --root-file arg (=/tmp/hmpRawDecodeResults)
                  Name of the Root file with the decoding
                  results.
  --fast-decode   Use the fast algorithm. (error 0.8%
```


### o2-hmpid-digits-to-raw-stream-workflow
Codes input o2::hmpid::Digits vector (HMP/DIGITS)  and (HMP/INTRECORD) stream into the binary HMPID raw file.

Display all options

```
o2-hmpid-digits-to-raw-stream-workflow --help full
```

Data processor options: HMP-WriteRawFromDigits:

```
  --out-file arg (=hmpidRaw)            name of the output file
  --order-events                        order the events time
  --skip-empty                          skip empty events
  --fixed-lenght                        fixed lenght packets = 8K bytes
```


### o2-hmpid-dump-digits-stream-workflow
Dumps the input o2::hmpid::Digits vector (HMP/DIGITS)  and (HMP/INTRECORD) stream to the stdout or in an ASCII file.

Display all options

```
o2-hmpid-dump-digits-stream-workflow --help full
```

Data processor options: HMP-DigitsDump:

```
  --out-file arg                        name of the output file
  --print                               print digits (default false )
```


### o2-hmpid-raw-to-digits-workflow
Write Raw File into a root formatted file

```
o2-hmpid-raw-to-digits-workflow --help full
```

Data processor options: HMPDigitWriter:

```
  --in-file arg (=hmpidRaw.raw)         name of the input Raw file
  --out-file arg (=hmpReco.root)        name of the output file
  --base-file arg (=hmpDecode)          base name for statistical output file
  --fast-decode                         Use the fast algorithm. (error 0.8%)
```

Example

```
[O2Suite/latest-o2] ~/Downloads/provaRec $> o2-hmpid-raw-to-digits-workflow --in-file test_full_flp1.raw --out-file hmpidReco.root --base-file /tmp/pippo -b
```

### o2-hmpid-digits-to-raw-workflow
Write raw files with the digits information contained in a root file

```
o2-hmpid-digits-to-raw-workflow
```

Data processor options: HMP-WriteRawFromRootFile:

```
  --outdir arg (=./)                    base dir for output file
  --file-for arg (=all)                 single file per: all,flp,link
  --outfile arg (=hmpid)                base name for output file
  --in-file arg (=hmpiddigits.root)     name of the input sim root file
  --dump-digits                         out the digits file in
                                        /tmp/hmpDumpDigits.dat
  --skip-empty                          skip empty events
```

Example

```
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-sim-serial -m HMP -n 20 -e TGeant4 -g pythia8hi
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-sim-digitizer-workflow --onlyDet HMP
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-hmpid-digits-to-raw-workflow --outdir ./ --in-file hmpiddigits.root --outfile hmpRawFromRoot --file-for all --dump-digits -b
```

in order to verify the write, the inverse decoding of raw file

```
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-hmpid-read-raw-file-stream-workflow --raw-file hmpRawFromRoot.raw -b | o2-hmpid-raw-to-digits-stream-workflow -b | o2-hmpid-dump-digits-stream-workflow --out-file /tmp/hmpDumpDigitsVerify.dat
```


### o2-hmpid-raw-to-pedestals-workflow
Write the Pedestals/Threshold files for the readout and registers Mean and Sigma in the CCDB

```
o2-hmpid-raw-to-pedestals-workflow --help full
```

Data processor options: HMP-DataDecoder:

```
  --files-basepath arg (=/tmp/hmpPedThr)
                                        Name of the Base Path of
                                        Pedestals/Thresholds files.
  --use-ccdb                            Register the Pedestals/Threshold values
                                        into the CCDB
  --ccdb-uri arg (=http://ccdb-test.cern.ch:8080)
                                        URI for the CCDB access.
  --pedestals-tag arg (=Latest)         The tag applied to this set of
                                        pedestals/threshold values
  --sigmacut arg (=4)                   Sigma values for the Thresholds
                                        calculation.
  --fast-decode                         Use the fast algorithm. (error 0.8%)
```

Example

```
o2-hmpid-read-raw-file-stream-workflow --raw-file ../hmpidRaw160.raw -b | o2-hmpid-raw-to-pedestals-workflow --sigmacut=2.5 --files-basepath /tmp/pippo -b --use-ccdb --pedestals-tag TEST3
```

this command produce in the ccdb a set of `TMatrixF` object one for each chamber

```
Subfolder
HMP/Pedestals/TEST3/Mean_0
HMP/Pedestals/TEST3/Mean_1
HMP/Pedestals/TEST3/Mean_2
HMP/Pedestals/TEST3/Mean_4
HMP/Pedestals/TEST3/Sigma_0
HMP/Pedestals/TEST3/Sigma_1
HMP/Pedestals/TEST3/Sigma_2
HMP/Pedestals/TEST3/Sigma_4
```

and in addition a set of Ped/Thr files in the `/tmp` folder

```
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_0.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_1.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_2.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_3.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_4.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_5.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_8.dat
-rw-r--r--  1 fap   wheel   92166 12 Feb 10:45 pippo_9.dat
```

the format of Ped/Thr is ASCII, one Hexadecimal Value for row **((Threshold & 0x001FF) << 9) | (Pedestal & 0x001FF)** , for a total of 15361 rows; the last value is the `End of File` indication (A0A0A)

```
0F674
1027E
1047F
1469F
13699
.
.
.
00000
00000
00000
00000
A0A0A
```
