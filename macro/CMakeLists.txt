# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# setup files to be installed (only ROOT macros for the moment) NOT using GLOB,
# as we should be mindful of what we install. if we have lot of files here, it's
# probably because most of them should be elsewhere in the first place ...

install(FILES CheckDigits_mft.C
              SetIncludePath.C
              analyzeHits.C
              duplicateHits.C
              migrateSimFiles.C
              analyzeDigitLabels.C
              analyzeOriginHits.C
              checkGeomOverlaps.C
              build_geometry.C
              checkTOFMatching.C
              compareTopologyDistributions.C
              eventDisplay.C
              initSimGeomAndField.C
              loadExtDepLib.C
              load_all_libs.C
              o2sim.C
              putCondition.C
              readEMCHits.C
              readITSDigits.C
              rootlogon.C
              runTPCRefit.C
              run_CRUDataSkimming_its.C
              run_calib_tof.C
              run_clus_itsSA.C
              run_clus_tof.C
              run_clus_tpc.C
              run_clus_emcal.C
              run_collect_calib_tof.C
              run_digi2raw_its.C
              run_digi2raw_mft.C
              run_digi2raw_tof.C
              run_cmp2digit_tof.C
              compareTOFDigits.C
              compareTOFClusters.C
              run_match_tof.C
              run_primary_vertexer_ITS.C
              run_rawdecoding_its.C
              run_rawdecoding_mft.C
              run_trac_ca_its.C
              run_trac_its.C
              run_trac_alice3.C
	      CreateBCPattern.C
        DESTINATION share/macro/)

# FIXME: a lot of macros that are here should really be elsewhere. Those which
# depends on a single subsystem should be located within that subsystem
# directory instead of in this global location. and this global location should
# be reserved for macros that use more than one subsystem ?

# FIXME: move to subsystem dir
o2_add_test_root_macro(CheckDigits_mft.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTBase O2::ITSMFTSimulation
                                             O2::MFTBase O2::MathUtils
                                             O2::SimulationDataFormat
                       LABELS mft)

o2_add_test_root_macro(analyzeHits.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTSimulation
                                             O2::TOFSimulation
                                             O2::EMCALBase
                                             O2::TRDSimulation
                                             O2::FT0Simulation
                                             O2::DataFormatsFV0
                                             O2::HMPIDBase
                                             O2::TPCSimulation
                                             O2::PHOSBase
                                             O2::FDDSimulation
                                             O2::MCHSimulation
                                             O2::MIDSimulation
					     O2::ZDCSimulation
					     O2::CPVBase)

o2_add_test_root_macro(duplicateHits.C
		PUBLIC_LINK_LIBRARIES O2::ITSMFTSimulation
		O2::TOFSimulation
		O2::EMCALBase
		O2::TRDSimulation
		O2::FT0Simulation
		O2::DataFormatsFV0
		O2::HMPIDBase
		O2::TPCSimulation
		O2::PHOSBase
		O2::FDDSimulation
		O2::MCHSimulation
		O2::MIDSimulation
		O2::ZDCSimulation
		O2::CPVBase)

o2_add_test_root_macro(migrateSimFiles.C
		PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats)

o2_add_test_root_macro(analyzeOriginHits.C
                       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                             O2::DetectorsCommonDataFormats)

o2_add_test_root_macro(build_geometry.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsPassive
                                             O2::Field
                                             O2::TPCSimulation
                                             O2::ITSSimulation
                                             O2::MFTSimulation
                                             O2::MCHSimulation
                                             O2::MIDSimulation
                                             O2::EMCALSimulation
                                             O2::TOFSimulation
                                             O2::TRDSimulation
                                             O2::FT0Simulation
                                             O2::FV0Simulation
                                             O2::FDDSimulation
                                             O2::HMPIDSimulation
                                             O2::PHOSSimulation
                                             O2::CPVSimulation
                                             O2::ZDCSimulation
                                             $<$<BOOL:${ENABLE_UPGRADES}>:O2::Alice3DetectorsPassive>
                                             $<$<BOOL:${ENABLE_UPGRADES}>:O2::ITS3Simulation>
                                             $<$<BOOL:${ENABLE_UPGRADES}>:O2::TRKSimulation>)

o2_add_test_root_macro(checkTOFMatching.C
                       PUBLIC_LINK_LIBRARIES O2::GlobalTracking
                                             O2::ReconstructionDataFormats
                                             O2::SimulationDataFormat
                                             O2::DataFormatsTOF)

# FIXME: move to subsystem dir
o2_add_test_root_macro(compareTopologyDistributions.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsITSMFT
                       LABELS its)

# FIXME: what's the point of this one ?
o2_add_test_root_macro(eventDisplay.C PUBLIC_LINK_LIBRARIES FairRoot::Base)

o2_add_test_root_macro(
  initSimGeomAndField.C
  PUBLIC_LINK_LIBRARIES O2::DataFormatsParameters O2::Field O2::DetectorsBase)

o2_add_test_root_macro(
        analyzeDigitLabels.C
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::ITSMFTSimulation
                                             O2::TOFSimulation
                                             O2::EMCALBase
                                             O2::TRDSimulation
                                             O2::FV0Simulation
                                             O2::FT0Simulation
                                             O2::DataFormatsFV0
                                             O2::HMPIDBase
                                             O2::TPCSimulation
                                             O2::PHOSBase
                                             O2::PHOSSimulation
                                             O2::FDDSimulation
                                             O2::MCHSimulation
                                             O2::MIDSimulation
					     O2::ZDCSimulation
					     O2::CPVBase
                                             O2::CPVSimulation
                                             O2::ZDCSimulation)

if(Geant4_FOUND AND BUILD_SIMULATION)
o2_add_test_root_macro(o2sim.C
                       PUBLIC_LINK_LIBRARIES O2::Generators
                                             O2::DetectorsPassive
                                             O2::Field
                                             O2::TPCSimulation
                                             O2::ITSSimulation
                                             O2::MFTSimulation
                                             O2::MCHSimulation
                                             O2::MIDSimulation
                                             O2::EMCALSimulation
                                             O2::TOFSimulation
                                             O2::TRDSimulation
                                             O2::FT0Simulation
                                             O2::FV0Simulation
                                             O2::FDDSimulation
                                             O2::HMPIDSimulation
                                             O2::PHOSSimulation
                                             O2::CPVSimulation
                                             O2::ZDCSimulation
                                             O2::CommonTypes
                                             O2::SimSetup
                                             O2::Steer)
endif()

# FIXME: move to subsystem dir + add includes if one wants to compile it...
# o2_add_test_root_macro( putCondition.C)
o2_add_test_root_macro(checkGeomOverlaps.C)

# FIXME: move to subsystem dir
o2_add_test_root_macro(readEMCHits.C
                       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                             O2::EMCALBase
                       LABELS emcal)

# FIXME: move to subsystem dir
o2_add_test_root_macro(readITSDigits.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsITSMFT
                                             O2::ITSMFTBase
                                             O2::SimulationDataFormat
                       LABELS its)

# FIXME: move to subsystem dir
o2_add_test_root_macro(runTPCRefit.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
                                             O2::ReconstructionDataFormats
                                             O2::SimulationDataFormat
                                             O2::TPCReconstruction
                       LABELS tpc)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_CRUDataSkimming_its.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTBase
                                             O2::ITSMFTReconstruction
                       LABELS its)

# FIXME: move to subsystem dir ?
o2_add_test_root_macro(run_calib_tof.C
                       PUBLIC_LINK_LIBRARIES O2::Field O2::DataFormatsParameters
                                             O2::DetectorsBase
                                             O2::GlobalTracking)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_itsSA.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                                             O2::ITSReconstruction
                                             O2::ITSMFTReconstruction
                                             O2::ITSMFTBase
                       LABELS its)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_tof.C
                       PUBLIC_LINK_LIBRARIES O2::TOFReconstruction O2::Framework O2::TOFBase
                                             O2::SimulationDataFormat O2::TOFCalibration O2::DataFormatsTOF
                       LABELS tof)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_tpc.C
                       PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                       LABELS tpc)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_clus_emcal.C
                       PUBLIC_LINK_LIBRARIES O2::EMCALReconstruction
                       LABELS emcal)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_collect_calib_tof.C
                       PUBLIC_LINK_LIBRARIES O2::GlobalTracking)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_digi2raw_its.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTBase
                                             O2::ITSMFTReconstruction
                       LABELS its)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_digi2raw_mft.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::ITSMFTBase
                                             O2::ITSMFTReconstruction
                       LABELS mft)

o2_add_test_root_macro(run_digi2raw_tof.C
                       PUBLIC_LINK_LIBRARIES O2::TOFSimulation
                                             O2::TOFReconstruction
                       LABELS tof)

o2_add_test_root_macro(run_cmp2digit_tof.C
                       PUBLIC_LINK_LIBRARIES O2::TOFBase
                                             O2::TOFReconstruction
                       LABELS tof)

o2_add_test_root_macro(compareTOFDigits.C
                       PUBLIC_LINK_LIBRARIES O2::TOFBase
                       LABELS tof)

o2_add_test_root_macro(compareTOFClusters.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTOF
                       LABELS tof)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_match_tof.C
                       PUBLIC_LINK_LIBRARIES O2::Field O2::DataFormatsParameters
                                             O2::DetectorsBase
                                             O2::GlobalTracking
                       LABELS tof)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_primary_vertexer_ITS.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsITSMFT
                                             O2::DataFormatsParameters
                                             O2::GPUTracking
                                             O2::SimulationDataFormat
                                             O2::ITSBase O2::ITStracking
					     O2::DetectorsBase
                       LABELS its COMPILE_ONLY)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_rawdecoding_its.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::CommonDataFormat
                       LABELS its)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_rawdecoding_mft.C
                       PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                             O2::DataFormatsITSMFT
                                             O2::CommonDataFormat
                       LABELS mft)

o2_add_test_root_macro(CreateBCPattern.C
                      PUBLIC_LINK_LIBRARIES O2::CommonDataFormat)

# FIXME: move to subsystem dir + check compilation o2_add_test_root_macro(
# run_rec_ca.C PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
# O2::DataFormatsITSMFT O2::DataFormatsParameters O2::DetectorsBase O2::Field
# O2::ITSBase O2::ITStracking O2::MathUtils O2::SimulationDataFormat)


# FIXME: move to subsystem dir
o2_add_test_root_macro(run_trac_ca_its.C
                       PUBLIC_LINK_LIBRARIES O2::GPUTracking
                       LABELS its COMPILE_ONLY)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_trac_its.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
                                             O2::DataFormatsITSMFT
                                             O2::DataFormatsParameters
                                             O2::DetectorsBase
                                             O2::Field
                                             O2::ITSBase
                                             O2::ITSReconstruction
                                             O2::ITStracking
                                             O2::MathUtils
                                             O2::SimulationDataFormat
                        LABELS its)

# FIXME: move to subsystem dir
o2_add_test_root_macro(run_trac_alice3.C
                       PUBLIC_LINK_LIBRARIES O2::GPUTracking O2::ITSMFTSimulation
                       LABELS its COMPILE_ONLY)

#
# NOTE: commented out until unit testing reenabled FIXME : re-enable or delete ?
#

# add_test_wrap(run_sim_tpc_TGeant3 ${CMAKE_BINARY_DIR}/macro/run_sim_tpc.sh 10
# \TGeant3\) Set_Tests_Properties(run_sim_tpc_TGeant3 PROPERTIES TIMEOUT 30)
# Set_Tests_Properties(run_sim_tpc_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION
# Macro finished succesfully)

# add_test_wrap(run_sim_tpc_TGeant4 ${CMAKE_BINARY_DIR}/macro/run_sim_tpc.sh 10
# \TGeant4\) Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES DEPENDS
# run_sim_tpc_TGeant3) Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES
# TIMEOUT 30) Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES
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
