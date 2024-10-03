# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(DataFormatsParameters
               SOURCES src/GRPObject.cxx
                       src/GRPLHCIFData.cxx
                       src/GRPECSObject.cxx
                       src/GRPMagField.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::CommonConstants
                                     O2::CommonTypes O2::CCDB
                                     O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(DataFormatsParameters
                          HEADERS include/DataFormatsParameters/ECSDataAdapters.h
                                  include/DataFormatsParameters/GRPObject.h
                                  include/DataFormatsParameters/GRPLHCIFData.h
                                  include/DataFormatsParameters/GRPECSObject.h
                                  include/DataFormatsParameters/GRPMagField.h
                          LINKDEF src/ParametersDataLinkDef.h)

o2_add_executable(simgrp-tool
                  COMPONENT_NAME grp
                  SOURCES src/GRPTool.cxx
                  PUBLIC_LINK_LIBRARIES Boost::program_options O2::DataFormatsParameters O2::SimConfig)

# note we are explicitely giving the LINKDEF parameter as the LinkDef does not
# follow the usual naming scheme [module]LinkDef.h
#
# * should be src/DataFormatParametersLinkDef.h
# * is src/ParametersDataLinkDef.h instead
