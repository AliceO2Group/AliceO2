<!-- doxy
\page refDetectorsMUONMCHIO IO
/doxy -->

# I/O tools for reading and writing MCH data in Root format

<!-- vim-markdown-toc GFM -->

* [Digit reader](#digit-reader)
* [Digit writer](#digit-writer)

<!-- vim-markdown-toc -->

## Digit reader

```shell
o2-mch-digits-reader-workflow
```

For historical reason the digit reader was targeted at simulated digits (for
which we have an extra branch containing MClabel information), so to read real
digit files, you'll unfortunately must use the `--disable-mc` option.

The digit file to be read can be specified with the `--mch-digit-infile` option.

For some expert usage you can also specify the data specifications that will be
used for the digits and the rofs messages (respectively `MCH/DIGITS` and
`MCH/DIGITROFS` by default) using the `--mch-output-digits-data-description` and
`-mch-output-digitrofs-data-description`options.

## Digit writer

```shell
o2-mch-digits-writer-workflow
```

The digit file to be written can be specified with the `--mch-digit-outfile` option.

For some expert usage you can also specify the data specifications that will be
used as sources for the digits and the rofs messages (respectively `MCH/DIGITS` and
`MCH/DIGITROFS` by default) using the `--mch-input-digits-data-description` and
`-mch-input-digitrofs-data-description`options.

