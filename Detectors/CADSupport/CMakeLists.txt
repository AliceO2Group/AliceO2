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

# Detectors/Base/src provides the bvh2 headers (bvh2_third_party.h, bvh2_extra_kernels.h) shared with O2Tessellated.
o2_add_library(CADSupport
               SOURCES src/O2BVHSurfaceSolid.cxx
                       src/O2BVHAssembly.cxx
                       src/O2SurfaceSolidIO.cxx
                       src/O2OverlapCheck.cxx
                       src/O2SolidHarness.cxx
                       src/CADGeometryUtils.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/Detectors/Base/src
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase ROOT::Geom)

o2_target_root_dictionary(CADSupport
                          HEADERS include/CADSupport/O2BVHSurfaceSolid.h
                                  include/CADSupport/O2BVHAssembly.h
                          LINKDEF src/CADSupportLinkDef.h)

o2_add_test(
  BVHSurfaceSolid
  SOURCES test/testBVHSurfaceSolid.cxx
  COMPONENT_NAME CADSupport
  PUBLIC_LINK_LIBRARIES O2::CADSupport ROOT::Geom ROOT::RIO
  LABELS cadsupport)

o2_add_test(
  BVHAssembly
  SOURCES test/testBVHAssembly.cxx
  COMPONENT_NAME CADSupport
  PUBLIC_LINK_LIBRARIES O2::CADSupport ROOT::Geom ROOT::RIO
  LABELS cadsupport)

o2_add_executable(
  solid-harness
  COMPONENT_NAME CADSupport
  SOURCES test/runSolidHarness.cxx
  IS_BENCHMARK
  PUBLIC_LINK_LIBRARIES O2::CADSupport ROOT::Geom ROOT::RIO
                        nlohmann_json::nlohmann_json)

# Overlap census of a placed geometry.
o2_add_executable(
  overlap
  COMPONENT_NAME CADSupport
  SOURCES test/runOverlapCensus.cxx
  IS_BENCHMARK
  PUBLIC_LINK_LIBRARIES O2::CADSupport ROOT::Geom ROOT::GeomPainter ROOT::RIO
                        nlohmann_json::nlohmann_json)
