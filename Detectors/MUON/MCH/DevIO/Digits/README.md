<!-- doxy
\page refDetectorsMUONMCHDevIODigits DigitsIO
/doxy -->

# Utilities to read and write MCH Digits in non-CTF formats (mostly for debug)

The MCH digits can be stored in binary format for debug purpose.

The file they are stored in can currently be of two different formats : with or
without associated ROFRecord. In both cases the digits themselves (currently)
have the same version (named D0). The file version number changes whenever the
 file structure changes or the format of the objects it contain changes.

| File version | Digit version | contains ROF | ROF version |  support  |
|:------------:|:-------------:|:------------:|:-----------:|:---------:|
|      0       |       0       |     no       |      0      |    yes    |
|      1       |       0       |     yes      |      0      |    yes    |
|      2       |       1       |     yes      |      0      |   planned |

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
  [--verbose]        print some basic information while processing
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
```

A device that reads digits with ids in Run2 convention and outputs digits with
 ids in Run3 convention. Note that expected input spec is `MCH/DIGITSRUN2` and 
 the output one is `MCH/DIGITS`
