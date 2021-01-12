# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(PWGDQCore
               SOURCES  src/VarManager.cxx
                        src/HistogramManager.cxx
                        src/AnalysisCut.cxx
                        src/AnalysisCompositeCut.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore)

o2_target_root_dictionary(PWGDQCore
                          HEADERS  include/PWGDQCore/AnalysisCut.h 
                          include/PWGDQCore/AnalysisCompositeCut.h 
                          include/PWGDQCore/VarManager.h 
                          include/PWGDQCore/HistogramManager.h
                          include/PWGDQCore/CutsLibrary.h 
                          include/PWGDQCore/HistogramsLibrary.h
                          LINKDEF src/PWGDQCoreLinkDef.h)
