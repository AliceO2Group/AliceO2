<!-- doxy
\page refDetectorsMUONWorkflow Workflows
/doxy -->

# MUON Workflows

<!-- vim-markdown-toc GFM -->

* [Matching](#matching)
* [Track writer](#track-writer)
* [MID chamber efficiency](#mid-chamber-efficiency)

<!-- vim-markdown-toc -->

## Matching

```shell
o2-muon-tracks-matcher-workflow
```

Take as input the lists of MCH tracks ([TrackMCH](../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)), MCH ROF records ([ROFRecord](../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)), MID tracks ([Track](../../../DataFormats/Detectors/MUON/MID/include/DataFormatsMID/Track.h)) and MID ROF records ([ROFRecord](../../../DataFormats/Detectors/MUON/MID/include/DataFormatsMID/ROFRecord.h)) in the current time frame, with the data descriptions "MCH/TRACKS", "MCH/TRACKROFS", "MID/TRACKS" and "MID/TRACKROFS", respectively. Send the list of matched tracks ([TrackMCHMID](../../../DataFormats/Reconstruction/include/ReconstructionDataFormats/TrackMCHMID.h)) in the time frame, with the data description "GLO/MTC_MCHMID".

Option `--disable-root-input` disables the reading of the input MCH and MID tracks from `mchtracks.root` and `mid-reco.root`, respectively.

Option `--disable-root-output` disables the writing of the ouput matched tracks to `muontracks.root`.

Option `--disable-mc` disables the reading, processing and writing of the MC label informations.

Option `--config "file.json"` or `--config "file.ini"` allows to change the matching parameters from a configuration file. This file can be either in JSON or in INI format, as described below:

* Example of configuration file in JSON format:

```json
{
    "MUONMatching": {
        "sigmaCut": "4."
    }
}
```

* Example of configuration file in INI format:

```ini
[MUONMatching]
sigmaCut=4.
```

Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change the matching parameters from the command line. The parameters changed from the command line will supersede the ones changed from a configuration file.

* Example of parameters changed from the command line:

```shell
--configKeyValues "MUONMatching.sigmaCut=4."
```

## Track writer

```shell
o2-muon-tracks-writer-workflow --outfile "muontracks.root"
```

Take as input the list of matched tracks ([TrackMCHMID](../../../DataFormats/Reconstruction/include/ReconstructionDataFormats/TrackMCHMID.h)) in the current time frame, with the data description "GLO/MTC_MCHMID", and write them in a root file. It is implemented using the generic [MakeRootTreeWriterSpec](../../../Framework/Utils/include/DPLUtils/MakeRootTreeWriterSpec.h) and thus offers the same options.

# MID chamber efficiency

This workflow allows to compute the MID chamber efficiency.
This is just an example since, eventually, the workflow should be rewritten in order to be able to run on AODs.

Usage:

```shell
o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet MID | o2-mid-reco-workflow --disable-mc | o2-mid-chamber-efficiency-workflow
```

The chamber efficiency can be estimated using only MID tracks that match MCH tracks:

```shell
o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet MCH,MID | o2-mid-reco-workflow --disable-mc | o2-mch-reco-workflow --disable-mc --disable-root-input --configKeyValues "MCHDigitFilter.timeOffset=126" | o2-muon-tracks-matcher-workflow --disable-mc --disable-root-input | o2-mid-chamber-efficiency-workflow --select-matched
```
