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
            CCDB/example/fill_local_ocdb.C
            DataFormats/simulation/test/checkStack.C
            Detectors/ITSMFT/ITS/macros/EVE/rootlogon.C
            Detectors/ITSMFT/ITS/macros/test/rootlogon.C
            Detectors/MUON/MCH/Simulation/macros/rootlogon.C
            Detectors/Passive/macro/PutFrameInTop.C
            Detectors/TPC/reconstruction/macro/addInclude.C
            Detectors/TPC/reconstruction/macro/getTPCTransformationExample.C
            GPU/GPUTracking/Merger/macros/checkPropagation.C
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldIts.C
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldTpc.C
            GPU/GPUTracking/Merger/macros/fitPolynomialFieldTrd.C
            GPU/GPUTracking/Standalone/tools/createGeo.C
            GPU/GPUTracking/Standalone/tools/createLUT.C
            GPU/GPUTracking/Standalone/tools/dump.C
            GPU/GPUTracking/TRDTracking/macros/checkDbgOutput.C
            GPU/TPCFastTransformation/macro/createTPCFastTransform.C
            GPU/TPCFastTransformation/macro/generateTPCDistortionNTuple.C
            GPU/TPCFastTransformation/macro/initTPCcalibration.C
            GPU/TPCFastTransformation/macro/loadlibs.C
            GPU/TPCFastTransformation/macro/moveTPCFastTransform.C
            Generators/share/external/hijing.C
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

