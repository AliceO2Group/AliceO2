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

# FairRootBase: the FairRoot simulation base classes, vendored into O2.
#
# These files are copied verbatim from ALICE's own FairRoot fork
# (alisw/FairRoot, v18.4.9-alice3) so that O2 no longer needs FairRoot as an
# external package. They remain LGPLv3 (see the per-file headers), which is
# compatible with O2's GPLv3.
#
# The classes deliberately keep their original names and header names in this
# first step: that makes the move a pure relocation with no call-site changes,
# so it can be validated as behaviour-preserving. Renaming them into an O2
# namespace, and pruning the parts O2 does not use, are follow-up steps.

# o2_target_root_dictionary requires paths relative to the current source dir
file(GLOB FAIRROOTBASE_SOURCES CONFIGURE_DEPENDS
     RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/src/*.cxx)
file(GLOB FAIRROOTBASE_HEADERS CONFIGURE_DEPENDS
     RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/include/*.h)

o2_add_library(FairRootBase
               TARGETVARNAME targetFairRootBase
               SOURCES ${FAIRROOTBASE_SOURCES}
               PUBLIC_LINK_LIBRARIES FairLogger::FairLogger
                                     MC::VMC
                                     Boost::serialization
                                     ROOT::Core
                                     ROOT::Geom
                                     # TGeoTrack, used by FairTrajFilter
                                     ROOT::GeomPainter
                                     ROOT::EG
                                     ROOT::Physics
                                     ROOT::Tree
                                     ROOT::RIO
                                     ROOT::Gdml)

# Vendored third-party code: do not fail the O2 build on its warnings.
target_compile_options(${targetFairRootBase} PRIVATE -w)

o2_target_root_dictionary(FairRootBase
                          HEADERS ${FAIRROOTBASE_HEADERS}
                          LINKDEF src/FairRootBaseLinkDef.h)

# These headers are included by bare name ("FairDetector.h") from O2 public
# headers, exactly as they were when they came from the FairRoot package, so
# they must be installed flat into <prefix>/include rather than under a
# per-target subdirectory (which is what o2_add_library would do).
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
