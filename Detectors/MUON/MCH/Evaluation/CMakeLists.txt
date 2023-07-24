#Copyright 2019 - 2020 CERN and copyright holders of ALICE O2.
#See https: // alice-o2.web.cern.ch/copyright for details of the copyright holders.
#All rights not expressly granted are reserved.
#
#This software is distributed under the terms of the GNU General Public
#License v3(GPL Version 3), copied verbatim in the file "COPYING".
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(MCHEvaluation
                 SOURCES
                   src /
               CompareTask.cxx
                 src /
               CompareTracks.cxx
                 src /
               Draw.cxx
                 src /
               ExtendedTrack.cxx
                 src /
               Histos.cxx
                 PUBLIC_LINK_LIBRARIES
                   O2::DataFormatsMCH
                     O2::MCHTracking
                       fmt::fmt)

  o2_add_executable(compare - tracks - worfklow SOURCES src / compare - tracks - workflow.cxx COMPONENT_NAME mch PUBLIC_LINK_LIBRARIES O2::MCHEvaluation)

    o2_target_root_dictionary(MCHEvaluation
                                HEADERS
                                  include /
                              MCHEvaluation / ExtendedTrack.h include / MCHEvaluation / Draw.h LINKDEF src / MCHEvaluationLinkDef.h)

      o2_add_executable(
        map_mch
          SOURCES src /
        map_mch.cxx
          COMPONENT_NAME mch
            PUBLIC_LINK_LIBRARIES O2::MCHMappingSegContour O2::MCHMappingImpl4 O2::MCHMappingInterface O2::MCHContour O2::MCHGeometryCreator O2::MCHGeometryTransformer O2::MCHConstants O2::MCHGlobalMapping
              Boost::program_options)