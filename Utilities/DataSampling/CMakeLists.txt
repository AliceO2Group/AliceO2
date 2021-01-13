# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# FIXME: the LinkDef should not be in the public area

o2_add_library(DataSampling
               SOURCES   src/DataSampling.cxx
                         src/DataSamplingConditionFactory.cxx
                         src/DataSamplingHeader.cxx
                         src/DataSamplingConditionCustom.cxx
                         src/DataSamplingConditionNConsecutive.cxx
                         src/DataSamplingConditionPayloadSize.cxx
                         src/DataSamplingConditionRandom.cxx
                         src/DataSamplingHeader.cxx
                         src/DataSamplingPolicy.cxx
                         src/DataSamplingReadoutAdapter.cxx
                         src/Dispatcher.cxx

  PUBLIC_LINK_LIBRARIES O2::Framework O2::DataSampling)

#o2_target_root_dictionary(
#  Mergers
#  HEADERS include/Mergers/MergeInterface.h
#  include/Mergers/CustomMergeableObject.h
#          include/Mergers/CustomMergeableTObject.h
#  LINKDEF include/Mergers/LinkDef.h)

# tests with input data

o2_data_file(COPY test/test_DataSampling.json test/test_DataSamplingEmpty.json DESTINATION tests)

o2_add_test(DataSampling NAME test_DataSampling_test_DataSampling
  SOURCES test/test_DataSampling.cxx
  COMPONENT_NAME DataSampling
  LABELS datasampling
  PUBLIC_LINK_LIBRARIES O2::DataSampling
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

foreach(t
  DataSamplingCondition
  DataSamplingHeader
  DataSamplingPolicy
  )

  # FIXME ? The NAME parameter of o2_add_test is only needed to help the current
  # o2.sh recipe. If the recipe is changed, those params can go away, if needed.

  o2_add_test(${t} NAME test_DataSampling_test_${t}
    SOURCES test/test_${t}.cxx
    COMPONENT_NAME DataSampling
    LABELS datasampling
    PUBLIC_LINK_LIBRARIES O2::DataSampling)
endforeach()

o2_data_file(COPY etc/exampleDataSamplingConfig.json DESTINATION etc)

o2_add_executable(standalone
  SOURCES src/dataSamplingStandalone.cxx
  COMPONENT_NAME DataSampling
  PUBLIC_LINK_LIBRARIES O2::Framework O2::DataSampling)

o2_add_executable(datasampling-pod-and-root
  SOURCES test/dataSamplingPodAndRoot.cxx
  COMPONENT_NAME DataSampling
PUBLIC_LINK_LIBRARIES O2::Framework O2::DataSampling)

o2_add_executable(datasampling-parallel
  SOURCES test/dataSamplingParallel.cxx
  COMPONENT_NAME DataSampling
PUBLIC_LINK_LIBRARIES O2::Framework O2::DataSampling)

o2_add_executable(datasampling-time-pipeline
  SOURCES test/dataSamplingTimePipeline.cxx
  COMPONENT_NAME DataSampling
  PUBLIC_LINK_LIBRARIES O2::Framework O2::DataSampling)

o2_add_dpl_workflow(datasampling-benchmark
  SOURCES test/dataSamplingBenchmark.cxx
  COMPONENT_NAME DataSampling
  PUBLIC_LINK_LIBRARIES O2::DataSampling)
