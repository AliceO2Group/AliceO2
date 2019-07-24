# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(GLFW_FOUND)
  set(GUI_SOURCES src/FrameworkGUIDebugger.cxx src/FrameworkGUIDevicesGraph.cxx
                  src/FrameworkGUIDeviceInspector.cxx
                  src/FrameworkGUIDataRelayerUsage.cxx src/PaletteHelpers.cxx)
else()
  set(GUI_SOURCES src/FrameworkDummyDebugger.cxx)
endif()

o2_add_library(Framework
               SOURCES src/AODReaderHelpers.cxx
                       ${GUI_SOURCES}
                       src/AnalysisHelpers.cxx
                       src/BoostOptionsRetriever.cxx
                       src/ChannelConfigurationPolicy.cxx
                       src/ChannelMatching.cxx
                       src/ChannelConfigurationPolicyHelpers.cxx
                       src/ChannelSpecHelpers.cxx
                       src/CommonDataProcessors.cxx
                       src/CompletionPolicy.cxx
                       src/CompletionPolicyHelpers.cxx
                       src/DispatchPolicy.cxx
                       src/ConfigParamsHelper.cxx
                       src/DDSConfigHelpers.cxx
                       src/DataAllocator.cxx
                       src/DataDescriptorMatcher.cxx
                       src/DataDescriptorQueryBuilder.cxx
                       src/DataProcessingDevice.cxx
                       src/DataProcessingHeader.cxx
                       src/DataProcessor.cxx
                       src/DataRelayer.cxx
                       src/DataSampling.cxx
                       src/DataSamplingConditionFactory.cxx
                       src/DataSamplingHeader.cxx
                       src/DataSamplingConditionCustom.cxx
                       src/DataSamplingConditionNConsecutive.cxx
                       src/DataSamplingConditionPayloadSize.cxx
                       src/DataSamplingConditionRandom.cxx
                       src/DataSamplingHeader.cxx
                       src/DataSamplingPolicy.cxx
                       src/DataSamplingReadoutAdapter.cxx
                       src/DataSpecUtils.cxx
                       src/DeviceMetricsInfo.cxx
                       src/DeviceSpec.cxx
                       src/DeviceSpecHelpers.cxx
                       src/Dispatcher.cxx
                       src/DriverControl.cxx
                       src/DriverInfo.cxx
                       src/FairMQDeviceProxy.cxx
                       src/FairMQResizableBuffer.cxx
                       src/FairOptionsRetriever.cxx
                       src/FreePortFinder.cxx
                       src/GraphvizHelpers.cxx
                       src/InputRecord.cxx
                       src/InputSpec.cxx
                       src/OutputSpec.cxx
                       src/Kernels.cxx
                       src/LifetimeHelpers.cxx
                       src/LocalRootFileService.cxx
                       src/LogParsingHelpers.cxx
                       src/MessageContext.cxx
                       src/Metric2DViewIndex.cxx
                       src/SimpleOptionsRetriever.cxx
                       src/O2ControlHelpers.cxx
                       src/OutputSpec.cxx
                       src/PropertyTreeHelpers.cxx
                       src/RCombinedDS.cxx
                       src/ReadoutAdapter.cxx
                       src/SimpleResourceManager.cxx
                       src/StreamOperators.cxx
                       src/TMessageSerializer.cxx
                       src/TableBuilder.cxx
                       src/TableConsumer.cxx
                       src/Task.cxx
                       src/TextControlService.cxx
                       src/Variant.cxx
                       src/WorkflowHelpers.cxx
                       src/WorkflowSerializationHelpers.cxx
                       src/WorkflowSpec.cxx
                       src/runDataProcessing.cxx
                       src/ExternalFairMQDeviceProxy.cxx
                       test/TestClasses.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES AliceO2::Common
                                     AliceO2::Configuration
                                     AliceO2::InfoLogger
                                     AliceO2::Monitoring
                                     CURL::libcurl
                                     FairMQ::FairMQ
                                     O2::CommonUtils
                                     O2::FrameworkFoundation
                                     O2::Headers
                                     O2::MemoryResources
                                     O2::PCG
                                     RapidJSON::RapidJSON
                                     arrow_shared
                                     ms_gsl::ms_gsl
                                     ROOT::ROOTDataFrame
                                     O2::DebugGUI
                                     O2::FrameworkLogger
                                     Boost::serialization)

o2_target_root_dictionary(Framework
                          HEADERS test/TestClasses.h
                          LINKDEF test/FrameworkCoreTestLinkDef.h)

if(GLFW_FOUND)
  foreach(t DebugGUISokol DebugGUIGL GUITests)
    o2_add_test(${t}
                SOURCES test/test_${t}.cxx
                COMPONENT_NAME Framework
                LABELS framework
                PUBLIC_LINK_LIBRARIES O2::Framework)
  endforeach()
  # FIXME: investigate those two failures at some point
  set_property(TEST Framework/Core/test/test_DebugGUISokol.cxx
               PROPERTY DISABLED True)
  set_property(TEST Framework/Core/test/test_DebugGUIGL.cxx
               PROPERTY DISABLED True)
endif()

foreach(t
        AlgorithmSpec
        AnalysisTask
        ASoA
        BoostOptionsRetriever
        CallbackRegistry
        ChannelSpecHelpers
        CompletionPolicy
        ConfigParamRegistry
        ContextRegistry
        DataDescriptorMatcher
        DataProcessorSpec
        DataRefUtils
        DataRelayer
        DataSamplingCondition
        DataSamplingHeader
        DataSamplingPolicy
        DeviceMetricsInfo
        DeviceSpec
        DeviceSpecHelpers
        ExternalFairMQDeviceProxy
        FairMQOptionsRetriever
        FairMQResizableBuffer
        FrameworkDataFlowToDDS
        Graphviz
        InfoLogger
        InputRecord
        Kernels
        LogParsingHelpers
        Parallel
        PtrHelpers
        Root2ArrowTable
        Services
        SuppressionGenerator
        TMessageSerializer
        TableBuilder
        TimeParallelPipelining
        TimesliceIndex
        TypeTraits
        Variants
        WorkflowHelpers
        WorkflowSerialization)

  # FIXME ? The NAME parameter of o2_add_test is only needed to help the current
  # o2.sh recipe. If the recipe is changed, those params can go away, if needed.

  o2_add_test(${t} NAME test_Framework_test_${t}
              SOURCES test/test_${t}.cxx
              COMPONENT_NAME Framework
              LABELS framework
              PUBLIC_LINK_LIBRARIES O2::Framework)
endforeach()

# tests with input data

o2_data_file(COPY test/test_DataSampling.json DESTINATION tests)

o2_add_test(DataSampling NAME test_Framework_test_DataSampling
            SOURCES test/test_DataSampling.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

# tests with a name not starting with test_...

o2_add_test(unittest_DataSpecUtils NAME test_Framework_unittest_DataSpecUtils
            SOURCES test/unittest_DataSpecUtils.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework)

o2_add_test(unittest_SimpleOptionsRetriever NAME
            test_Framework_unittest_SimpleOptionsRetriever
            SOURCES test/unittest_SimpleOptionsRetriever.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework)

# benchmarks

foreach(b
        ContextRegistry
        DataDescriptorMatcher
        DataRelayer
        DeviceMetricsInfo
        InputRecord
        TableBuilder)
  o2_add_test(benchmark_${b} NAME test_Framework_benchmark_${b}
              SOURCES test/benchmark_${b}.cxx
              COMPONENT_NAME Framework
              LABELS framework benchmark
              PUBLIC_LINK_LIBRARIES O2::Framework benchmark::benchmark)
endforeach()

# #####################################################@

foreach(w
        BoostSerializedProcessing
        CallbackService
        CustomGUIGL
        CustomGUISokol
        DanglingInputs
        DanglingOutputs
        DataAllocator
        StaggeringWorkflow
        Forwarding
        ParallelPipeline
        ParallelProducer
        SimpleDataProcessingDevice01
        SimpleRDataFrameProcessing
        SimpleStatefulProcessing01
        SimpleStringProcessing
        SimpleTimer
        SingleDataSource)
  o2_add_test(${w} NAME test_Framework_test_${w}
              SOURCES test/test_${w}.cxx
              COMPONENT_NAME Framework
              LABELS framework workflow
              PUBLIC_LINK_LIBRARIES O2::Framework
              TIMEOUT 30
              NO_BOOST_TEST
              COMMAND_LINE_ARGS --run)
endforeach()

# TODO: DanglingInput test not working for the moment
#    [ERROR] Unable to relay part.
#    [WARN] Incoming data is already obsolete, not relaying.
set_property(TEST test_Framework_test_DanglingInputs
             PROPERTY DISABLED TRUE)

# TODO: investigate the problem with the two unit tests, maybe setup of the CI environment
# assertion fired
#     X11: The DISPLAY environment variable is missing
#     glfw-3.2.1/src/window.c:579: glfwGetFramebufferSize: Assertion `window != ((void *)0)' failed.
set_property(TEST test_Framework_test_CustomGUIGL
             PROPERTY DISABLED TRUE)
set_property(TEST test_Framework_test_CustomGUISokol
             PROPERTY DISABLED TRUE)

# TODO: investigate the problem and re-enable
set_property(TEST test_Framework_test_BoostSerializedProcessing
             PROPERTY DISABLED TRUE)


# specific tests which needs command line options
o2_add_test(
  ProcessorOptions NAME test_Framework_test_ProcessorOptions
  SOURCES test/test_ProcessorOptions.cxx
  COMPONENT_NAME Framework
  LABELS framework workflow
  TIMEOUT 60
  PUBLIC_LINK_LIBRARIES O2::Framework
  NO_BOOST_TEST
  COMMAND_LINE_ARGS
    --global-config require-me --run
    # Note: the group switch makes process consumer parse only the group
    arguments --consumer
    "--global-config consumer-config --local-option hello-aliceo2 --a-boolean3 --an-int2 20 --a-double2 22."
  )
