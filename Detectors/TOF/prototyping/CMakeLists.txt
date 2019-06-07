o2_add_test_root_macro(checkRotation.C PUBLIC_LINK_LIBRARIES TOFBase)

o2_add_test_root_macro(convertTreeTo02object.C PUBLIC_LINK_LIBRARIES
                       ReconstructionDataFormats)

o2_add_test_root_macro(drawTOFgeometry.C PUBLIC_LINK_LIBRARIES DetectorsPassive
                       TOFSimulation)

o2_add_test_root_macro(findLabels.C
                       PUBLIC_LINK_LIBRARIES
                       SimulationDataFormat
                       ReconstructionDataFormats
                       DataFormatsTPC
                       DataFormatsITS
                       DataFormatsTOF
                       TOFBase)

o2_add_test_root_macro(findTOFclusterFromLabel.C
                       PUBLIC_LINK_LIBRARIES
                       DataFormatsTOF
                       SimulationDataFormat
                       DataFormatsTPC
                       ReconstructionDataFormats
                       CommonDataFormat
                       TOFBase)
