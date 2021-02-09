<!-- doxy
\page refHMPworkflow HMP workflow
/doxy -->

# DPL workflows for the HMPID

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