# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(AnalysisDataModel
               SOURCES src/dummy.cxx
                       src/ParamBase.cxx
                       src/DetectorResponse.cxx
                       src/PIDTOF.cxx
                       src/PIDTPC.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework O2::ReconstructionDataFormats O2::TPCSimulation
              )


o2_target_root_dictionary(AnalysisDataModel
                          HEADERS include/PIDBase/ParamBase.h
                                  include/PIDBase/DetectorResponse.h
                                  include/PID/PIDTOF.h
                                  include/PID/TOFReso.h
                                  include/PID/PIDTPC.h
                                  include/PID/BetheBloch.h
                                  include/PID/TPCReso.h
                          LINKDEF src/AnalysisDataModelLinkDef.h
                         )


o2_add_executable(dump-aod-data-model
                  SOURCES src/dumpDataModel.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                 )

o2_add_executable(pidparam-tof-reso
                  SOURCES src/handleParamTOFReso.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                 )
