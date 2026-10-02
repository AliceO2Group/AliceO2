# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(ACTSInterface
               TARGETVARNAME targetName
               SOURCES src/TrackingGeometryManager.cxx
                       src/SurfaceIndexMap.cxx
                       src/MagneticFieldAdapter.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                                     O2::DetectorsCommonDataFormats
                                     O2::Field
                                     O2::Framework
                                     ROOT::Geom
                                     Acts::Core
                                     Acts::PluginRoot
                                     Acts::PluginJson)

target_compile_definitions(${targetName} PUBLIC O2_WITH_ACTS)
