<!-- doxy
\page refHMPworkflow HMP workflow
/doxy -->

<<<<<<< HEAD
# DPL workflows for the HMPID  v.0.4

## HMPID DPL processors

* `hmpid-read-raw-file` reads data from ReadOut Raw File and outputs a stream of RDH6 pages as HMP/RAWDATA stream
* `hmpid-raw-to-digits` decodes the input raw pages stream (HMP/RAWDATA) and produces a vector of o2::hmpid::Digits (HMP/DIGITS)
* `hmpid-write-raw-from-digits` codes input o2::hmpid::Digits vector (HMP/DIGITS) into the binary HMPID raw file
* `hmpid-dump-digits` dumps the input o2::hmpid::Digits vector (HMP/DIGITS) to the stdout or in a ASCII file

### Workflow example
The input is a HMPID rawfile and after the decoding the result is a reconstructed rawfile.

```
 o2-hmpid-read-raw-file-workflow --raw-file test_full_flp1.raw -b | o2-hmpid-raw-to-digits-workflow -b | o2-hmpid-write-raw-from-digits-workflow --out-file /tmp/pippo -b
```

This reads the `test_full_flp1.raw` file and after the decodeing produce a couple of raw files in the /tmp/ folder that are prefixed with `pippo`



### o2-hmpid-read-raw-file-workflow
Reads data from ReadOut Raw File and outputs a stream of RDH6 pages as HMP/RAWDATA stream.

Display all options

```
o2-hmpid-read-raw-file-workflow --help full
```

Data processor options: HMP-ReadRawFile:

```
--raw-file arg                        Raw input file name
```

### o2-hmpid-raw-to-digits-workflow
Decodes the input raw pages stream (HMP/RAWDATA) and produces a vector of o2::hmpid::Digits (HMP/DIGITS).

Display all options

```
o2-hmpid-raw-to-digits-workflow --help full
```

Data processor options: HMP-DataDecoder:

```
  --root-file arg (=/tmp/hmpRawDecodeResults)
                                        Name of the Root file with the decoding
                                        results.
```


### o2-hmpid-write-raw-from-digits-workflow
Codes input o2::hmpid::Digits vector (HMP/DIGITS) into the binary HMPID raw file.

Display all options

```
o2-hmpid-write-raw-from-digits-workflow --help full
```

Data processor options: HMP-WriteRawFromDigits:

```
  --out-file arg (=hmpidRaw)            name prefix of the two output files
  --order-events                        order the events in ascending time
  --skip-empty                          skip empty events
  --fixed-lenght                        fixed lenght packets = 8K bytes
```


### o2-hmpid-dump-digits-workflow
Dumps the input o2::hmpid::Digits vector (HMP/DIGITS) to the stdout or in a ASCII file.

Display all options

```
o2-hmpid-dump-digits-workflow --help full
```

Data processor options: HMP-DigitsDump:

```
  --out-file arg                        name of the output file
  --print                               print digits (default false )
```


### o2-hmpid-write-root-from-digits-workflow
Write the digit stream into a root formatted file

```
o2-hmpid-write-root-from-digit-workflow --help full
```

Data processor options: HMPDigitWriter:

```
  --outfile arg (=hmpiddigits.root)     Name of the output file
  --treename arg (=o2sim)               Name of tree
  --treetitle arg (=o2sim)              Title of tree
  --nevents arg (=-1)                   Number of events to execute
  --terminate arg (=process)            Terminate the 'process' or 'workflow'
```

Example

```
[O2Suite/latest-o2] ~/Downloads/provaRec $> o2-hmpid-read-raw-file-workflow --raw-file test_full_flp1.raw -b | o2-hmpid-raw-to-digits-workflow -b | o2-hmpid-write-root-from-digits-workflow -b
```

### o2-hmpid-write-raw-from-root-workflow
Write raw files with the digits information contained in a root file

```
o2-hmpid-write-raw-from-root-workflow
```

Data processor options: HMP-WriteRawFromRootFile:

```
  --hmp-raw-outdir arg (=./)            base dir for output file
  --hmp-raw-outfile arg (=hmpReadOut)   base name for output file
  --hmp-raw-perlink                     produce one file per link
  --hmp-raw-perflp                      produce one raw file per FLPs
  --in-file arg (=hmpiddigits.root)     name of the input sim root file
  --dump-digits                         out the digits file in /tmp/hmpDumpDigits.dat
  --hmp-skip-empty                      skip empty events
  --start-value-enumeration arg (=0)    initial value for the enumeration
  --end-value-enumeration arg (=-1)     final value for the enumeration
  --step-value-enumeration arg (=1)     step between one value and the other
```

Example

```
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-sim-serial -m HMP -n 20 -e TGeant4 -g pythia8hi
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-sim-digitizer-workflow --onlyDet HMP
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-hmpid-write-raw-from-root-workflow --in-file hmpiddigits.root --hmp-raw-outfile hmpRawFromRoot --dump-digits -b
```

in order to verify the write, the inverse decoding of raw file

```
[O2Suite/latest-o2] ~/Downloads/provaRec $>o2-hmpid-read-raw-file-workflow --raw-file hmpRawFromRoot.raw -b | o2-hmpid-raw-to-digits-workflow -b | o2-hmpid-dump-digits-workflow --out-file /tmp/hmpDumpDigitsVerify.dat
```

=======
# DPL workflows for the HMPID  v.0.3

## HMPID DPL processors

* `hmpid-read-raw-file` reads data from ReadOut Raw File and outputs a stream of RDH6 pages as HMP/RAWDATA stream
* `hmpid-raw-to-digits` decodes the input raw pages stream (HMP/RAWDATA) and produces a vector of o2::hmpid::Digits (HMP/DIGITS)
* `hmpid-write-raw-from-digits` codes input o2::hmpid::Digits vector (HMP/DIGITS) into the binary HMPID raw file
* `hmpid-dump-digits` dumps the input o2::hmpid::Digits vector (HMP/DIGITS) to the stdout or in a ASCII file

### Workflow example
The input is a HMPID rawfile and after the decoding the result is a reconstructed rawfile.

```
 o2-hmpid-read-raw-file-workflow --raw-file test_full_flp1.raw -b | o2-hmpid-raw-to-digits-workflow -b | o2-hmpid-write-raw-from-digits-workflow --out-file /tmp/pippo -b
```

This reads the `test_full_flp1.raw` file and after the decodeing produce a couple of raw files in the /tmp/ folder that are prefixed with `pippo`



### o2-hmpid-read-raw-file-workflow
Reads data from ReadOut Raw File and outputs a stream of RDH6 pages as HMP/RAWDATA stream.

Display all options

```
o2-hmpid-read-raw-file-workflow --help full
```

Data processor options: HMP-ReadRawFile:

```
--raw-file arg                        Raw input file name
```

### o2-hmpid-raw-to-digits-workflow
Decodes the input raw pages stream (HMP/RAWDATA) and produces a vector of o2::hmpid::Digits (HMP/DIGITS).

Display all options

```
o2-hmpid-raw-to-digits-workflow --help full
```

Data processor options: HMP-DataDecoder:

```
  --root-file arg (=/tmp/hmpRawDecodeResults)
                                        Name of the Root file with the decoding
                                        results.
```


### o2-hmpid-write-raw-from-digits-workflow
Codes input o2::hmpid::Digits vector (HMP/DIGITS) into the binary HMPID raw file.

Display all options

```
o2-hmpid-write-raw-from-digits-workflow --help full
```

Data processor options: HMP-WriteRawFromDigits:

```
  --out-file arg (=hmpidRaw)            name prefix of the two output files
  --order-events                        order the events in ascending time
  --skip-empty                          skip empty events
  --fixed-lenght                        fixed lenght packets = 8K bytes
```


### o2-hmpid-dump-digits-workflow
Dumps the input o2::hmpid::Digits vector (HMP/DIGITS) to the stdout or in a ASCII file.

Display all options

```
o2-hmpid-dump-digits-workflow --help full
```

Data processor options: HMP-DigitsDump:

```
  --out-file arg                        name of the output file
  --print                               print digits (default false )
```


### o2-hmpid-write-root-from-digits-workflow
Write the digit stream into a root formatted file

```
o2-hmpid-write-root-from-digit-workflow --help full
```

Data processor options: HMPDigitWriter:

```
  --outfile arg (=hmpiddigits.root)     Name of the output file
  --treename arg (=o2sim)               Name of tree
  --treetitle arg (=o2sim)              Title of tree
  --nevents arg (=-1)                   Number of events to execute
  --terminate arg (=process)            Terminate the 'process' or 'workflow'
```

Example

```
[O2Suite/latest-o2] ~/Downloads/provaRec $> o2-hmpid-read-raw-file-workflow --raw-file test_full_flp1.raw -b | o2-hmpid-raw-to-digits-workflow -b | o2-hmpid-write-root-from-digits-workflow -b
```
>>>>>>> refs/heads/dev

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
```

Example

```
o2-hmpid-read-raw-file-workflow --raw-file test_full_flp1.raw -b | o2-hmpid-raw-to-pedestals-workflow --sigmacut=2.5 --files-basepath /tmp/pippo -b --use-ccdb --pedestals-tag TEST3
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
