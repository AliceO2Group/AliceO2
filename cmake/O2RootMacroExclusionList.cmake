# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

include(O2GetListOfMacros)

#
# List of macros that are allowed (hopefully temporarily) to escape testing.
#

list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST
            DataFormats/simulation/test/checkStack.C
            Detectors/ITSMFT/ITS/macros/EVE/rootlogon.C
            Detectors/ITSMFT/ITS/macros/test/rootlogon.C
            Detectors/MUON/MCH/Simulation/macros/rootlogon.C
            Detectors/Passive/macro/PutFrameInTop.C
            Detectors/TPC/reconstruction/macro/addInclude.C
            Detectors/TPC/reconstruction/macro/getTPCTransformationExample.C
            Detectors/EMCAL/calib/macros/ReadTestBadChannelMap_CCDBApi.C
            GPU/GPUTracking/Merger/macros/checkPropagation.C # Needs AliRoot AliExternalTrackParam
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldIts.C # Needs AliRoot AliMagF
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldTpc.C # Needs AliRoot AliMagF
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldTrd.C # Needs AliRoot AliMagF
            GPU/GPUTracking/Standalone/tools/dump.C # Needs AliRoot ALiHLTSystem
            GPU/GPUTracking/TRDTracking/macros/checkDbgOutput.C # Needs AliRoot TStatToolkit
            GPU/TPCFastTransformation/macro/createTPCFastTransform.C # Needs AliTPCCalibDB
            GPU/TPCFastTransformation/macro/generateTPCDistortionNTupleAliRoot.C # Needs AliTPCCalibDB
            GPU/TPCFastTransformation/macro/initTPCcalibration.C # Needs AliTPCCalibDB
            GPU/TPCFastTransformation/macro/loadlibs.C # Special macro
            GPU/TPCFastTransformation/macro/moveTPCFastTransform.C # Relies on initTPCcalibration.C
            Generators/share/external/hijing.C
	    Generators/share/external/QEDepem.C
            macro/SetIncludePath.C
            macro/loadExtDepLib.C
            macro/load_all_libs.C
            macro/putCondition.C
            macro/rootlogon.C)

if(NOT BUILD_SIMULATION)
  # some complete sub_directories are not added to the build when not building
  # simulation, so the corresponding o2_add_test_root_macro won't be called at
  # all
  o2_get_list_of_macros(${CMAKE_SOURCE_DIR}/macro macros)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST ${macros})
  o2_get_list_of_macros(${CMAKE_SOURCE_DIR}/Detectors/gconfig macros)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST ${macros})

   list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST Generators/share/external/QEDLoader.C)
endif()

if(NOT pythia6_FOUND)
  list(APPEND O2_ROOT_MACRO_EXCLUSION_LIST Generators/share/external/pythia6.C)
endif()

list(REMOVE_DUPLICATES O2_ROOT_MACRO_EXCLUSION_LIST)

# check exclusion list contains only existing macros
foreach(m ${O2_ROOT_MACRO_EXCLUSION_LIST})
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${m})
    message(FATAL_ERROR "Exclusion list contains a non-existing macro : ${m}")
  endif()
endforeach()
