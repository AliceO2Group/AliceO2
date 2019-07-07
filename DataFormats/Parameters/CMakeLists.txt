# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsParameters
               SOURCES src/GRPObject.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::CommonConstants
                                     O2::CommonTypes
                                     O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(DataFormatsParameters
                          HEADERS include/DataFormatsParameters/GRPObject.h
                          LINKDEF src/ParametersDataLinkDef.h)
# note we are explicitely giving the LINKDEF parameter as the LinkDef does not
# follow the usual naming scheme [module]LinkDef.h
#
# * should be src/DataFormatParametersLinkDef.h
# * is src/ParametersDataLinkDef.h instead
