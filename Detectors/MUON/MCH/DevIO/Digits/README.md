<!-- doxy
\page refDetectorsMUONMCHDevIODigits DigitsIO
/doxy -->

<!-- vim-markdown-toc GFM -->

* [Utilities to read and write MCH Digits in non-CTF formats (mostly for debug)](#utilities-to-read-and-write-mch-digits-in-non-ctf-formats-mostly-for-debug)
  * [Standalone programs (i.e. not DPL devices)](#standalone-programs-ie-not-dpl-devices)
    * [Digit file dumper](#digit-file-dumper)
  * [DPL devices](#dpl-devices)
    * [Digit file reader (aka sampler) device](#digit-file-reader-aka-sampler-device)
    * [Digit writer (aka sink) device](#digit-writer-aka-sink-device)
    * [Digit ID converter (from Run2 to Run3)](#digit-id-converter-from-run2-to-run3)
    * [Digit random generator](#digit-random-generator)

<!-- vim-markdown-toc -->
# Utilities to read and write MCH Digits in non-CTF formats (mostly for debug)

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

## Standalone programs (i.e. not DPL devices)

### Digit file dumper

```shell
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

## DPL devices

Those are (partial) DPL workflows to be used to form bigger workflows
 using pipes.

### Digit file reader (aka sampler) device

Reads a binary file containing digits. The binary format should be embedded
in the file directly. If the reader cannot identify the format it will assume
it is the V0 version, which might or not might be the case...

```
o2-mch-digits-file-reader-workflow
  --infile arg       input file name
  [--max-nof-tfs]    max number of timeframes to process
  [--first-tf]       first timeframe to process
  [--max-nof-rofs]   max number of ROFs to process
  [--repack-rofs]    number of rofs to repack into a timeframe
                     (aka min number of rofs per timeframe)
  [--print-digits]   print digits
  [--print-tfs]      print number of digits and rofs per tf
```

### Digit writer (aka sink) device

```
o2-mch-digits-file-writer-workflow
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

```
o2-mch-digits-r23-workflow
  [--verbose]        print ids being converted
```

A device that reads digits with ids in Run2 convention and outputs digits with
 ids in Run3 convention. Note that expected input spec is `MCH/DIGITSRUN2` and 
 the output one is `MCH/DIGITS`

### Digit random generator

```
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

```
o2-mch-digits-random-generator-workflow -b
  --max-nof-tfs 10 
  --nof-rofs-per-tf 128 
  --occupancy 0.01 | 
o2-mch-digits-writer-workflow -b 
  --print-tfs 
  --binary-file-format 1 
  --outfile digits.v1.out
```

```
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
