#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_library(FDDRaw
               SOURCES src/DataBlockFDD.cxx src/DigitBlockFDD.cxx src/RawReaderFDDBase.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonDataFormat O2::Headers O2::DataFormatsFDD O2::FITRaw)
