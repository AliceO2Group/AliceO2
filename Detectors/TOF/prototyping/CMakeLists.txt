# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test_root_macro(checkRotation.C
                       PUBLIC_LINK_LIBRARIES O2::TOFBase
                       LABELS tof)

o2_add_test_root_macro(convertTreeTo02object.C
                       PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
		       			     O2::DataFormatsTOF 
                       LABELS tof)

o2_add_test_root_macro(drawTOFgeometry.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsPassive
                                             O2::TOFSimulation
                       LABELS tof)

o2_add_test_root_macro(findLabels.C
                       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                             O2::ReconstructionDataFormats
                                             O2::DataFormatsTPC
                                             O2::DataFormatsITS
                                             O2::DataFormatsTOF
                                             O2::TOFBase
                       LABELS tof)

o2_add_test_root_macro(findTOFclusterFromLabel.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTOF
                                             O2::SimulationDataFormat
                                             O2::DataFormatsTPC
                                             O2::ReconstructionDataFormats
                                             O2::CommonDataFormat
                                             O2::TOFBase
                       LABELS tof)
