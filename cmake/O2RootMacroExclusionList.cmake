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

include_guard()

include(O2GetListOfMacros)

#
# List of macros that are allowed (hopefully temporarily) to escape testing.
#

list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST
            Detectors/ITSMFT/ITS/macros/test/CheckLUtime.C # temporary exclude until fix for full Clusters elimination
            Detectors/ITSMFT/ITS/macros/test/dictionary_integrity_test.C     # temporary exclude until fix for full Clusters elimination
            Detectors/MUON/MCH/Geometry/Test/rootlogon.C
            Detectors/Passive/macro/PutFrameInTop.C
            Detectors/TPC/reconstruction/macro/addInclude.C
            Detectors/TPC/reconstruction/macro/getTPCTransformationExample.C
            Detectors/TRD/base/macros/OCDB2CCDB.C
            Detectors/TRD/base/macros/OCDB2CCDBTrapConfig.C
            Detectors/TRD/base/macros/Readocdb.C
            Detectors/TRD/base/macros/PrintTrapConfig.C
            Detectors/TRD/base/macros/TestTrapSim.C
            Detectors/TRD/macros/convertRun2ToRun3Digits.C
            Detectors/TRD/simulation/macros/CheckTRDFST.C
            Detectors/gconfig/g4Config.C
            Detectors/TRD/macros/ParseTrapRawOutput.C
            Detectors/EMCAL/calib/macros/ReadTestBadChannelMap_CCDBApi.C
            GPU/GPUTracking/display/filterMacros/TRDCandidate.C
            GPU/GPUTracking/display/filterMacros/hasTRD.C
            GPU/GPUTracking/display/filterMacros/filterGPUTrack.C
            GPU/GPUTracking/display/filterMacros/filterTPCTrack.C
            GPU/GPUTracking/Merger/macros/checkPropagation.C # Needs AliRoot AliExternalTrackParam
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldIts.C # Needs AliRoot AliMagF
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldTpc.C # Needs AliRoot AliMagF
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldTrd.C # Needs AliRoot AliMagF
            GPU/GPUTracking/Standalone/tools/dump.C # Needs AliRoot ALiHLTSystem
            GPU/GPUTracking/Standalone/tools/dumpTRDClusterMatrices.C # Needs AliRoot AliCDBManager, AliGeomManager and AliTRDgeometry
            GPU/GPUTracking/TRDTracking/macros/checkDbgOutput.C # Needs AliRoot TStatToolkit
            GPU/TPCFastTransformation/alirootMacro/createTPCFastTransform.C # Needs AliTPCCalibDB
            GPU/TPCFastTransformation/alirootMacro/generateTPCDistortionNTupleAliRoot.C # Needs AliTPCCalibDB
            GPU/TPCFastTransformation/alirootMacro/initTPCcalibration.C # Needs AliTPCCalibDB
            GPU/TPCFastTransformation/devtools/loadlibs.C # Special macro
            GPU/TPCFastTransformation/alirootMacro/moveTPCFastTransform.C # Relies on initTPCcalibration.C
            GPU/GPUTracking/TRDTracking/macros/run_trd_tracker.C # Not yet ready
            Detectors/TOF/prototyping/ConvertRun2CalibrationToO2.C
            Generators/share/external/hijing.C
            Generators/share/external/QEDepem.C
            Generators/share/external/GenCosmics.C
            macro/SetIncludePath.C
            macro/loadExtDepLib.C
            macro/load_all_libs.C
            macro/putCondition.C
            macro/rootlogon.C
            Detectors/FIT/FT0/calibration/macros/makeChannelOffsetCalibObjectInCCDB.C)


if(NOT BUILD_SIMULATION)
  # some complete sub_directories are not added to the build when not building
  # simulation, so the corresponding o2_add_test_root_macro won't be called at
  # all
  o2_get_list_of_macros(${CMAKE_SOURCE_DIR}/macro macros)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST ${macros})
  o2_get_list_of_macros(${CMAKE_SOURCE_DIR}/Detectors/gconfig macros)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST ${macros})
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST Generators/share/external/QEDLoader.C)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST Generators/share/external/GenCosmicsLoader.C)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST Generators/share/egconfig/pythia8_userhooks_charm.C)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST Generators/share/external/trigger_mpi.C)
endif()

if(NOT ENABLE_UPGRADES)
  # exclude all the macros found under Detectors/Upgrades directory
  o2_get_list_of_macros(${CMAKE_SOURCE_DIR}/Detectors/Upgrades upgradeMacros)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST ${upgradeMacros})
endif()

list(REMOVE_DUPLICATES O2_ROOT_MACRO_EXCLUSION_LIST)

# check exclusion list contains only existing macros
foreach(m ${O2_ROOT_MACRO_EXCLUSION_LIST})
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${m})
    message(FATAL_ERROR "Exclusion list contains a non-existing macro : ${m}")
  endif()
endforeach()
