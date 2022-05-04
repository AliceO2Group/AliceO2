<!-- doxy
\page refDetectorsMUONWorkflow Workflows
/doxy -->

# MUON Workflows

<!-- vim-markdown-toc GFM -->

* [Matching](#matching)
* [Track writer](#track-writer)

<!-- vim-markdown-toc -->

## Matching

```shell
o2-muon-tracks-matcher-workflow
```

Take as input the lists of MCH tracks ([TrackMCH](../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/TrackMCH.h)), MCH ROF records ([ROFRecord](../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)), MID tracks ([Track](../../../DataFormats/Detectors/MUON/MID/include/DataFormatsMID/Track.h)) and MID ROF records ([ROFRecord](../../../DataFormats/Detectors/MUON/MID/include/DataFormatsMID/ROFRecord.h)) in the current time frame, with the data descriptions "MCH/TRACKS", "MCH/TRACKROFS", "MID/TRACKS" and "MID/TRACKROFS", respectively. Send the list of matched tracks ([TrackMCHMID](../../../DataFormats/Reconstruction/include/ReconstructionDataFormats/TrackMCHMID.h)) in the time frame, with the data description "GLO/MCHMID".

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

Take as input the list of matched tracks ([TrackMCHMID](../../../DataFormats/Reconstruction/include/ReconstructionDataFormats/TrackMCHMID.h)) in the current time frame, with the data description "GLO/MCHMID", and write them in a root file. It is implemented using the generic [MakeRootTreeWriterSpec](../../../Framework/Utils/include/DPLUtils/MakeRootTreeWriterSpec.h) and thus offers the same options.
