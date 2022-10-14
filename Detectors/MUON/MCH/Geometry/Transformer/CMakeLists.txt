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

o2_add_library(
  MCHGeometryTransformer
  SOURCES src/Transformations.cxx src/VolumePaths.cxx src/ClusterTransformerSpec.cxx
  PUBLIC_LINK_LIBRARIES ROOT::Geom O2::MathUtils Microsoft.GSL::GSL
                        RapidJSON::RapidJSON O2::Framework O2::DetectorsBase
                        O2::DataFormatsMCH
  PRIVATE_LINK_LIBRARIES O2::MCHConstants)

o2_add_executable(
  convert-geometry
  SOURCES src/convert-geometry.cxx
  PUBLIC_LINK_LIBRARIES ROOT::Geom Microsoft.GSL::GSL Boost::program_options
                        O2::MCHGeometryTransformer
  COMPONENT_NAME mch)

o2_add_executable(
        clusters-transformer-workflow
        SOURCES src/clusters-transformer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
           O2::DetectorsBase
           O2::Framework
           O2::MCHBase
           O2::MCHGeometryTransformer
        )

