#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_library(ZDCReconstruction
               SOURCES src/CTFCoder.cxx
               src/CTFHelper.cxx
               PUBLIC_LINK_LIBRARIES O2::ZDCBase
                                     O2::DataFormatsZDC
                                     O2::DetectorsRaw
                                     AliceO2::InfoLogger
                     O2::rANS
                     Microsoft.GSL::GSL)

# o2_target_root_dictionary(ZDCReconstruction
#                       HEADERS )
