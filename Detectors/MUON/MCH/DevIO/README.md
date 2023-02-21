<!-- doxy
\page refDetectorsMUONMCHDevIO DevIO
/doxy -->

# I/O tools for reading and writing MCH data in custom formats

This is mostly for debug and/or usage with Run2 (converted) data.

<!-- vim-markdown-toc GFM -->

* [Digit I/O](#digit-io)
  * [Digit sampler](#digit-sampler)
  * [Digit sink](#digit-sink)
  * [Digit ID converter (from Run2 to Run3)](#digit-id-converter-from-run2-to-run3)
  * [Digit random generator](#digit-random-generator)
  * [Digit file dumper](#digit-file-dumper)
* [Precluster I/O](#precluster-io)
  * [Precluster sink](#precluster-sink)

<!-- vim-markdown-toc -->
## Digit I/O

The MCH digits can be stored in binary format for debug purpose.

The file they are stored in can currently be of two different formats : with or
without associated ROFRecord. In both cases the digits themselves (currently)
have the same version (named D0). The file version number changes whenever the
 file structure changes or the format of the objects it contain changes.

| File version | Digit version | contains ROF | ROF version |  support  |
|:------------:|:-------------:|:------------:|:-----------:|:---------:|
|      0       |       0       |     no       |     (0)     |    yes    |
|      1       |       0       |     yes      |      1      |    yes    |
|      2       |       1       |     yes      |      1      |   planned |
|      X       |       X       |     yes      |      X      |   xxxxxxx |

### Digit sampler

Reads a binary file containing digits. The binary format should be embedded
in the file directly. If the reader cannot identify the format it will assume
it is the V0 version, which might or not might be the case...

```shell
o2-mch-digits-sampler-workflow
  --infile arg       input file name
  [--max-nof-tfs]    max number of timeframes to process
  [--first-tf]       first timeframe to process
  [--max-nof-rofs]   max number of ROFs to process
  [--repack-rofs]    number of rofs to repack into a timeframe
                     (aka min number of rofs per timeframe)
  [--print-digits]   print digits
  [--print-tfs]      print number of digits and rofs per tf
```

### Digit sink

```shell
o2-mch-digits-sink-workflow
  [--outfile] arg (=digits.out)       output file name
  [--max-nof-tfs] arg (=2147483647)   max number of timeframes to process
  [--first-tf] arg (=0)               first timeframe to process
  [--no-file]                         no output to file
  [--binary-file-format] arg (=v1)    digit binary format to use
  [--print-digits]                    print digits
  [--print-tfs]                       print number of digits and rofs per tf
  [--txt]                             output digits in text format
  [--max-size] arg (=2147483647)      max output size (in KB)
```

Take as input a list of digits in timeframes (and optionally their associated
Orbits) and dump them into a file (binary or text if `--txt`
option is used) or to screen (if `--no-file` option is used)

### Digit ID converter (from Run2 to Run3)

```shell
o2-mch-digits-r23-workflow
  [--verbose]        print ids being converted
```

A device that reads digits with ids in Run2 convention and outputs digits with
 ids in Run3 convention. Note that expected input spec is `MCH/DIGITSRUN2` and
 the output one is `MCH/DIGITS`

### Digit random generator

```shell
o2-mch-digits-random-generator-workflow
  [--first-tf] arg (=0)               first timeframe to generate
  [--max-nof-tfs] arg (=2147483647)   max number of timeframes to generate
  [--nof-rofs-per-tf]                 number of ROFs to generate in each timeframe
  [--occupancy]                       fraction of pads to generate digits for
  [--print-digits]                    print digits
  [--print-tfs]                       print number of digits and rofs per tf
```

Generate a fixed number of digits (equal to the total number of MCH pads
times the occupancy) per ROFRecord for a fixed number of ROFRecord per time frames,
 for a fixed number of timeframes.

Note that the occupancy **must** be strictly positive and less than or equal to 1.

Example of use : generate digits (with 1% occupancy, 128 ROFs per TF, 10 TFs) and
dump them into debug binary form :

```shell
o2-mch-digits-random-generator-workflow -b
  --max-nof-tfs 10
  --nof-rofs-per-tf 128
  --occupancy 0.01 |
o2-mch-digits-writer-workflow -b
  --print-tfs
  --binary-file-format 1
  --outfile digits.v1.out
```

```shell
$ ls -alrth digits.v1.out
-rw-r--r--  1 laurent  staff   258M 15 avr 10:30 digits.v1.out

$ o2-mch-digits-file-dumper --infile digits.v1.out -c -d
[ file version 1 digit version 0 size 20 rof version 1 size 16 hasRof true run2ids false ] formatWord 1224998065220435759
nTFs 10 nROFs 1280 nDigits 13506560
```

The generated digit file can then be injected into other workflows using
`o2-mch-digits-file-reader-workflow` described above.
(digits do not _need_ to be written to disc, the `o2-mch-digits-random-generator-workflow`
 can of course also be used as the first stage of a multi device workflow
 using pipes).

### Digit file dumper

The digit file dumper is not, in contrast to all the program above, a DPL worfklow, but a "regular" simple program, that dumps digit information on screen.

```shell
o2-mch-digits-dumper
options:
  -h [ --help ]          produce help message
  -i [ --infile ] arg    input file name
  -c [ --count ]         count items (rofs, tfs, etc...)
  -d [ --describe ]      describe file format
  -p [ --print-digits ]  print digits
  -t [ --print-tfs ]     print number of digits and rofs per tf
  --max-nof-tfs arg      max number of timeframes to process
  --first-tf arg         first timeframe to process
```

## Precluster I/O

### Precluster sink

```shell
o2-mch-preclusters-sink-workflow --outfile "preclusters.out"
```

Take as input the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the current time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction, with the data description "PRECLUSTERS", "PRECLUSTERDIGITS" and "PRECLUSTERROFS", respectively, and write them event-by-event in the binary file `preclusters.out` with the following format for each event:

* number of preclusters (int)
* number of associated digits (int)
* list of preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h))
* list of associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h))

Option `--txt` allows to write the preclusters in the output file in text format.

Option `--useRun2DigitUID` allows to convert the run3 pad ID stored in the digit data member mPadID into a digit UID in run2 format.

