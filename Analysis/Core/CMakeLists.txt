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
               SOURCES  src/CorrelationContainer.cxx
                        src/TrackSelection.cxx
                        src/VarManager.cxx
                        src/HistogramManager.cxx
                        src/AnalysisCut.cxx
                        src/AnalysisCompositeCut.cxx
                        src/TriggerAliases.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisTools)

o2_target_root_dictionary(AnalysisCore
              HEADERS include/Analysis/CorrelationContainer.h
                      include/Analysis/TrackSelection.h
                      include/Analysis/TrackSelectionDefaults.h
                      include/Analysis/VarManager.h
                      include/Analysis/HistogramManager.h
                      include/Analysis/AnalysisCut.h
                      include/Analysis/AnalysisCompositeCut.h
                      include/Analysis/TriggerAliases.h
                      include/Analysis/MC.h
              LINKDEF src/AnalysisCoreLinkDef.h)

if(FastJet_FOUND)
o2_add_library(AnalysisJets
               SOURCES  src/JetFinder.cxx
               PUBLIC_LINK_LIBRARIES O2::AnalysisCore FastJet::FastJet FastJet::Contrib)

o2_target_root_dictionary(AnalysisJets
              HEADERS include/Analysis/JetFinder.h
              LINKDEF src/AnalysisJetsLinkDef.h)
endif()
