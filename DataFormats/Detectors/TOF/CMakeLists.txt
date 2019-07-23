# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsTOF
               SOURCES src/Cluster.cxx
                       src/CalibInfoTOFshort.cxx
                       src/CalibInfoTOF.cxx
                       src/CalibLHCphaseTOF.cxx
                       src/CalibTimeSlewingParamTOF.cxx
               PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
                                     Boost::serialization)

o2_target_root_dictionary(DataFormatsTOF
                          HEADERS include/DataFormatsTOF/Cluster.h
                                  include/DataFormatsTOF/CalibInfoTOFshort.h
                                  include/DataFormatsTOF/CalibInfoTOF.h
                                  include/DataFormatsTOF/CalibLHCphaseTOF.h
                                  include/DataFormatsTOF/CalibTimeSlewingParamTOF.h)
