# setup files to be installed (only ROOT macros for the moment) NOT using GLOB,
# as we should be mindful of what we install. if we have lot of files here, it's
# probably because most of them should be elsewhere in the first place ...

install(FILES CheckClusters_mft.C
              CheckDigits_mft.C
              SetIncludePath.C
              analyzeHits.C
              build_geometry.C
              checkTOFMatching.C
              compareTopologyDistributions.C
              convertClusterToClusterHardware.C
              eventDisplay.C
              initSimGeomAndField.C
              loadExtDepLib.C
              load_all_libs.C
              o2sim.C
              putCondition.C
              readEMCHits.C
              readITSDigits.C
              rootlogon.C
              runCATrackingClusterNative.C
              run_CRUDataSkimming_its.C
              run_calib_tof.C
              run_clus_itsSA.C
              run_clus_mftSA.C
              run_clus_tof.C
              run_clus_tpc.C
              run_collect_calib_tof.C
              run_digi.C
              run_digi2raw_its.C
              run_digi_mft.C
              run_digi_tof.C
              run_match_TPCITS.C
              run_match_tof.C
              run_primary_vertexer_ITS.C
              run_rawdecoding_its.C
              run_sim.C
              run_sim_emcal.C
              run_sim_mft.C
              run_sim_pythia8hi.C
              run_sim_tof.C
              run_sim_tpc.C
              run_trac_ca_its.C
              run_trac_its.C
              run_trac_mft.C
        DESTINATION share/macro/)

# FIXME: a lot of macros that are here should really be elsewhere. Those which
# depends on a single subsystem should be located within that subsystem
# directory instead of in this global location. and this global location should
# be reserved for macros that use more than one subsystem ?

# FIXME: move to subsystem dir
o2_add_test_root_macro(CheckClusters_mft.C
                       PUBLIC_LINK_LIBRARIES DataFormatsITSMFT ITSMFTSimulation
                                             MFTBase MathUtils
                                             SimulationDataFormat)

# FIXME: move to subsystem dir
o2_add_test_root_macro(CheckDigits_mft.C
                       PUBLIC_LINK_LIBRARIES ITSMFTBase ITSMFTSimulation MFTBase
                                             MathUtils SimulationDataFormat)

o2_add_test_root_macro(analyzeHits.C
                       PUBLIC_LINK_LIBRARIES ITSMFTSimulation
                                             TOFSimulation
                                             EMCALBase
                                             TRDSimulation
                                             T0Simulation
                                             DataFormatsFITV0
                                             HMPIDBase
                                             TPCSimulation
                                             PHOSBase
                                             FDDSimulation)

o2_add_test_root_macro(build_geometry.C
                       PUBLIC_LINK_LIBRARIES DetectorsPassive
                                             Field
                                             TPCSimulation
                                             ITSSimulation
                                             MFTSimulation
                                             MCHSimulation
                                             MIDSimulation
                                             EMCALSimulation
                                             TOFSimulation
                                             TRDSimulation
                                             T0Simulation
                                             V0Simulation
                                             FDDSimulation
                                             HMPIDSimulation
                                             PHOSSimulation
                                             CPVSimulation
                                             ZDCSimulation)

o2_add_test_root_macro(checkTOFMatching.C
                       PUBLIC_LINK_LIBRARIES GlobalTracking
                                             ReconstructionDataFormats
                                             SimulationDataFormat)

# FIXME: move to subsystem dir
o2_add_test_root_macro(compareTopologyDistributions.C
                       PUBLIC_LINK_LIBRARIES DataFormatsITSMFT)
# FIXME: move to subsystem dir
o2_add_test_root_macro(convertClusterToClusterHardware.C
                       PUBLIC_LINK_LIBRARIES SimulationDataFormat DataFormatsTPC
                                             TPCBase TPCReconstruction)

# FIXME: what's the point of this one ?
o2_add_test_root_macro(eventDisplay.C PUBLIC_LINK_LIBRARIES FairRoot::Base)

o2_add_test_root_macro(initSimGeomAndField.C
                       PUBLIC_LINK_LIBRARIES DataFormatsParameters Field)

o2_add_test_root_macro(o2sim.C
                       PUBLIC_LINK_LIBRARIES Generators
                                             DetectorsPassive
                                             Field
                                             TPCSimulation
                                             ITSSimulation
                                             MFTSimulation
                                             MCHSimulation
                                             MIDSimulation
                                             EMCALSimulation
                                             TOFSimulation
                                             TRDSimulation
                                             T0Simulation
                                             V0Simulation
                                             FDDSimulation
                                             HMPIDSimulation
                                             PHOSSimulation
                                             CPVSimulation
                                             ZDCSimulation
                                             CommonTypes
                                             SimSetup)

# FIXME: move to subsystem dir + add includes if one wants to compile it...
# o2_add_test_root_macro( putCondition.C)

# FIXME: move to subsystem dir
o2_add_test_root_macro(readEMCHits.C
                       PUBLIC_LINK_LIBRARIES SimulationDataFormat EMCALBase)

# FIXME: move to subsystem dir
o2_add_test_root_macro(readITSDigits.C
                       PUBLIC_LINK_LIBRARIES DataFormatsITSMFT ITSMFTBase
                                             SimulationDataFormat)

# FIXME: move to subsystem dir
o2_add_test_root_macro(runCATrackingClusterNative.C
                       PUBLIC_LINK_LIBRARIES DataFormatsTPC
                                             ReconstructionDataFormats
                                             SimulationDataFormat
                                             TPCReconstruction)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_CRUDataSkimming_its.C
                       PUBLIC_LINK_LIBRARIES ITSMFTReconstruction
                                             DataFormatsITSMFT ITSMFTBase
                                             ITSMFTReconstruction)

# FIXME: move to subsystem dir ?
o2_add_test_root_macro(run_calib_tof.C
                       PUBLIC_LINK_LIBRARIES Field DataFormatsParameters
                                             DetectorsBase GlobalTracking)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_itsSA.C
                       PUBLIC_LINK_LIBRARIES DetectorsBase ITSReconstruction
                                             ITSMFTReconstruction ITSMFTBase)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_mftSA.C
                       PUBLIC_LINK_LIBRARIES DetectorsBase MFTReconstruction
                                             ITSMFTReconstruction)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_tof.C PUBLIC_LINK_LIBRARIES TOFReconstruction)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_tpc.C PUBLIC_LINK_LIBRARIES TPCReconstruction)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_collect_calib_tof.C
                       PUBLIC_LINK_LIBRARIES GlobalTracking)

o2_add_test_root_macro(run_digi.C
                       PUBLIC_LINK_LIBRARIES FairRoot::Base
                                             FairLogger::FairLogger)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_digi2raw_its.C
                       PUBLIC_LINK_LIBRARIES ITSMFTReconstruction
                                             DataFormatsITSMFT ITSMFTBase
                                             ITSMFTReconstruction)

# FIXME: move to subsystem dir
o2_add_test_root_macro(
  run_digi_mft.C
  PUBLIC_LINK_LIBRARIES DataFormatsParameters MFTSimulation)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_digi_tof.C PUBLIC_LINK_LIBRARIES TOFSimulation)

o2_add_test_root_macro(run_match_TPCITS.C
                       PUBLIC_LINK_LIBRARIES Field
                                             DataFormatsParameters
                                             DetectorsBase
                                             DataFormatsTPC
                                             TPCReconstruction
                                             GlobalTracking
                                             ITSMFTBase)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_match_tof.C
                       PUBLIC_LINK_LIBRARIES Field DataFormatsParameters
                                             DetectorsBase GlobalTracking)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_primary_vertexer_ITS.C
                       PUBLIC_LINK_LIBRARIES DataFormatsITSMFT
                                             DataFormatsParameters
                                             SimulationDataFormat ITSBase
                                             ITStracking)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_rawdecoding_its.C
                       PUBLIC_LINK_LIBRARIES ITSMFTReconstruction
                                             DataFormatsITSMFT CommonDataFormat)

# FIXME: move to subsystem dir + check compilation o2_add_test_root_macro(
# run_rec_ca.C PUBLIC_LINK_LIBRARIES DetectorsCommonDataFormats
# DataFormatsITSMFT DataFormatsParameters DetectorsBase Field ITSBase
# ITStracking MathUtils SimulationDataFormat)

o2_add_test_root_macro(run_sim.C
                       PUBLIC_LINK_LIBRARIES DetectorsPassive
                                             Field
                                             ITSBase
                                             ITSMFTBase
                                             ITSSimulation
                                             TPCSimulation)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_sim_emcal.C
                       PUBLIC_LINK_LIBRARIES DetectorsPassive Field
                                             EMCALSimulation)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_sim_mft.C
                       PUBLIC_LINK_LIBRARIES DetectorsPassive Field MFTBase
                                             MFTSimulation)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_sim_pythia8hi.C
                       PUBLIC_LINK_LIBRARIES Field DetectorsPassive
                                             TPCSimulation Generators)

o2_add_test_root_macro(run_sim_tof.C
                       PUBLIC_LINK_LIBRARIES Field DetectorsPassive Generators
                                             TPCSimulation TOFSimulation)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_sim_tpc.C
                       PUBLIC_LINK_LIBRARIES Field DetectorsPassive Generators
                                             TPCSimulation)

# FIXME: move to subsystem dir + check how to deal with GPU dependencies
# o2_add_test_root_macro( run_trac_ca_its.C)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_trac_its.C
                       PUBLIC_LINK_LIBRARIES DetectorsCommonDataFormats
                                             DataFormatsITSMFT
                                             DataFormatsParameters
                                             DetectorsBase
                                             Field
                                             ITSBase
                                             ITSReconstruction
                                             MathUtils
                                             SimulationDataFormat)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_trac_mft.C
                       PUBLIC_LINK_LIBRARIES Field MFTReconstruction)

#
# NOTE: commented out until unit testing reenabled

# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_sim_tpc.C)

# add_test_wrap(run_sim_tpc_TGeant3 ${CMAKE_BINARY_DIR}/macro/run_sim_tpc.sh 10
# \TGeant3\) Set_Tests_Properties(run_sim_tpc_TGeant3 PROPERTIES TIMEOUT 30)
# Set_Tests_Properties(run_sim_tpc_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION
# Macro finished succesfully)

# add_test_wrap(run_sim_tpc_TGeant4 ${CMAKE_BINARY_DIR}/macro/run_sim_tpc.sh 10
# \TGeant4\) Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES DEPENDS
# run_sim_tpc_TGeant3) Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)

# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_digi_tpc.C)

# add_test_wrap(run_digi_tpc_TGeant3 ${CMAKE_BINARY_DIR}/macro/run_digi_tpc.sh
# 10 \TGeant3\) Set_Tests_Properties(run_digi_tpc_TGeant3 PROPERTIES DEPENDS
# run_sim_TGeant3) Set_Tests_Properties(run_digi_tpc_TGeant3 PROPERTIES TIMEOUT
# 30) Set_Tests_Properties(run_digi_tpc_TGeant3 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)

# add_test_wrap(run_digi_tpc_TGeant4 ${CMAKE_BINARY_DIR}/macro/run_digi_tpc.sh
# 10 \TGeant4\) Set_Tests_Properties(run_digi_tpc_TGeant4 PROPERTIES DEPENDS
# run_sim_tpc_TGeant4) Set_Tests_Properties(run_digi_tpc_TGeant4 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(run_digi_tpc_TGeant4 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)

# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_clusterer.C)

# add_test_wrap(run_clusterer_TGeant3 ${CMAKE_BINARY_DIR}/macro/run_clusterer.sh
# 10 \TGeant3\) Set_Tests_Properties(run_clusterer_TGeant3 PROPERTIES DEPENDS
# run_digi_tpc_TGeant3) Set_Tests_Properties(run_clusterer_TGeant3 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(run_clusterer_TGeant3 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)

# add_test_wrap(comp_clusterer_TGeant3
# ${CMAKE_BINARY_DIR}/macro/compare_cluster.sh 10 \TGeant3\)
# Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES DEPENDS
# run_clusterer_TGeant3) Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)

# add_test_wrap(run_clusterer_TGeant4 ${CMAKE_BINARY_DIR}/macro/run_clusterer.sh
# 10 \TGeant4\) Set_Tests_Properties(run_clusterer_TGeant4 PROPERTIES DEPENDS
# run_digi_tpc_TGeant4) Set_Tests_Properties(run_clusterer_TGeant4 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(run_clusterer_TGeant4 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)

# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/test_GBTFrame.C)
# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/test_fixedPoint.C)
# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/compare_cluster.C)

# add_test_wrap(comp_clusterer_TGeant3
# ${CMAKE_BINARY_DIR}/macro/compare_cluster.sh 10 \TGeant3\)
# Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES DEPENDS
# run_clusterer_TGeant3) Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES
# PASS_REGULAR_EXPRESSION Macro finished succesfully)
# add_test_wrap(test_GBTFrame ${CMAKE_BINARY_DIR}/macro/test_GBTFrame.sh)
# Set_Tests_Properties(test_GBTFrame PROPERTIES TIMEOUT 30)
# Set_Tests_Properties(test_GBTFrame PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully)

# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/load_all_libs.C)

# #ITS tests with G3 configure_file(${CMAKE_SOURCE_DIR}/macro/run_sim_its.sh
# ${CMAKE_BINARY_DIR}/macro/run_sim_its.sh)
# configure_file(${CMAKE_SOURCE_DIR}/macro/run_sim_its.C
# ${CMAKE_BINARY_DIR}/macro/run_sim_its.C)
# configure_file(${CMAKE_SOURCE_DIR}/macro/SetIncludePath.C
# ${CMAKE_BINARY_DIR}/macro/SetIncludePath.C)

# add_test_wrap(NAME run_sim_its_G3 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_sim_its.sh 10 TGeant3)
# set_tests_properties(run_sim_its_G3 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_sim_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully)

# configure_file(${CMAKE_SOURCE_DIR}/macro/run_digi_its.sh
# ${CMAKE_BINARY_DIR}/macro/run_digi_its.sh)
# configure_file(${CMAKE_SOURCE_DIR}/macro/run_digi_its.C
# ${CMAKE_BINARY_DIR}/macro/run_digi_its.C)

# add_test_wrap(NAME run_digi_its_G3 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_digi_its.sh  10 TGeant3)
# set_tests_properties(run_digi_its_G3 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_digi_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully) set_tests_properties(run_digi_its_G3 PROPERTIES DEPENDS
# run_sim_its_G3)

# configure_file(${CMAKE_SOURCE_DIR}/macro/run_clus_its.sh
# ${CMAKE_BINARY_DIR}/macro/run_clus_its.sh)
# configure_file(${CMAKE_SOURCE_DIR}/macro/run_clus_its.C
# ${CMAKE_BINARY_DIR}/macro/run_clus_its.C)

# add_test_wrap(NAME run_clus_its_G3 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_clus_its.sh 10 TGeant3)
# set_tests_properties(run_clus_its_G3 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_clus_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully) set_tests_properties(run_clus_its_G3 PROPERTIES DEPENDS
# run_digi_its_G3)

# configure_file(${CMAKE_SOURCE_DIR}/macro/run_trac_its.sh
# ${CMAKE_BINARY_DIR}/macro/run_trac_its.sh)
# configure_file(${CMAKE_SOURCE_DIR}/macro/run_trac_its.C
# ${CMAKE_BINARY_DIR}/macro/run_trac_its.C)

# add_test_wrap(NAME run_trac_its_G3 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_trac_its.sh 10 TGeant3)
# set_tests_properties(run_trac_its_G3 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_trac_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully) set_tests_properties(run_trac_its_G3 PROPERTIES DEPENDS
# run_clus_its_G3)

# #ITS tests with G4

# add_test_wrap(NAME run_sim_its_G4 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_sim_its.sh 10 TGeant4)
# set_tests_properties(run_sim_its_G4 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_sim_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully)

# add_test_wrap(NAME run_digi_its_G4 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_digi_its.sh  10 TGeant4)
# set_tests_properties(run_digi_its_G4 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_digi_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully) set_tests_properties(run_digi_its_G4 PROPERTIES DEPENDS
# run_sim_its_G4)

# add_test_wrap(NAME run_clus_its_G4 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_clus_its.sh 10 TGeant4)
# set_tests_properties(run_clus_its_G4 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_clus_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully) set_tests_properties(run_clus_its_G4 PROPERTIES DEPENDS
# run_digi_its_G4)

# add_test_wrap(NAME run_trac_its_G4 COMMAND
# ${CMAKE_BINARY_DIR}/macro/run_trac_its.sh 10 TGeant4)
# set_tests_properties(run_trac_its_G4 PROPERTIES TIMEOUT 30)
# set_tests_properties(run_trac_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully) set_tests_properties(run_trac_its_G4 PROPERTIES DEPENDS
# run_clus_its_G4)

# GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/load_all_libs.C)
# add_test_wrap(load_all_libs ${CMAKE_BINARY_DIR}/macro/load_all_libs.sh)
# Set_Tests_Properties(load_all_libs PROPERTIES TIMEOUT 30)
# Set_Tests_Properties(load_all_libs PROPERTIES PASS_REGULAR_EXPRESSION Macro
# finished succesfully.)
