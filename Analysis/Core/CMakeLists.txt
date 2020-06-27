# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(AnalysisCore
               SOURCES  src/StepTHn.cxx
                        src/CorrelationContainer.cxx
                        src/TrackSelection.cxx
                        src/VarManager.cxx
                        src/HistogramManager.cxx
                        src/RunToTimestamp.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel)

o2_target_root_dictionary(AnalysisCore
              HEADERS include/Analysis/StepTHn.h
                      include/Analysis/CorrelationContainer.h
                      include/Analysis/TrackSelection.h
                      include/Analysis/VarManager.h
                      include/Analysis/HistogramManager.h
                      include/Analysis/RunToTimestamp.h
                      include/Analysis/MC.h
              LINKDEF src/AnalysisCoreLinkDef.h)

o2_add_executable(makerun2timestamp
            COMPONENT_NAME AnalysisCore
            SOURCES src/InsertNewRunToTimestamp.cxx
            PUBLIC_LINK_LIBRARIES O2::AnalysisCore O2::CCDB)

