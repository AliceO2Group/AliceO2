o2_add_test_root_macro(CheckClusterShape.C
                       PUBLIC_LINK_LIBRARIES
                       ITSBase
                       ITSMFTSimulation
                       SimulationDataFormat
                       LABELS
                       its)

o2_add_test_root_macro(CheckClusters.C
                       PUBLIC_LINK_LIBRARIES
                       ITSBase
                       DataFormatsITSMFT
                       ITSMFTSimulation
                       MathUtils
                       SimulationDataFormat
                       LABELS
                       its)

o2_add_test_root_macro(CheckDigits.C
                       PUBLIC_LINK_LIBRARIES
                       ITSBase
                       ITSMFTBase
                       ITSMFTSimulation
                       MathUtils
                       SimulationDataFormat
                       DetectorsBase
                       LABELS
                       its)

o2_add_test_root_macro(CheckLUtime.C
                       PUBLIC_LINK_LIBRARIES
                       ITSMFTReconstruction
                       DataFormatsITSMFT
                       LABELS
                       its)

o2_add_test_root_macro(CheckTopologies.C
                       PUBLIC_LINK_LIBRARIES
                       MathUtils
                       ITSBase
                       ITSMFTReconstruction
                       ITSMFTSimulation
                       DataFormatsITSMFT
                       SimulationDataFormat
                       LABELS
                       its)

o2_add_test_root_macro(CheckTracks.C
                       PUBLIC_LINK_LIBRARIES
                       SimulationDataFormat
                       DataFormatsITS
                       DataFormatsITSMFT
                       LABELS
                       its)

o2_add_test_root_macro(DisplayTrack.C
                       PUBLIC_LINK_LIBRARIES
                       ITSBase
                       DataFormatsITSMFT
                       ITSMFTSimulation
                       DataFormatsITS
                       MathUtils
                       SimulationDataFormat
                       LABELS
                       its)

o2_add_test_root_macro(dictionary_integrity_test.C
                       PUBLIC_LINK_LIBRARIES
                       DataFormatsITSMFT
                       ITSMFTReconstruction
                       LABELS
                       its)

o2_add_test_root_macro(run_buildTopoDict_its.C
                       PUBLIC_LINK_LIBRARIES
                       MathUtils
                       ITSBase
                       ITSMFTReconstruction
                       DataFormatsITSMFT
                       ITSMFTSimulation
                       SimulationDataFormat
                       LABELS
                       its)
