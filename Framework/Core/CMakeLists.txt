# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Given GCC 7.3 does not provide std::filesystem we use Boost instead
# Drop this once we move to GCC 8.2+
# if (NOT __APPLE__)
  set(BOOST_FILESYSTEM Boost::filesystem)
# endif()

o2_add_library(Framework
               SOURCES src/AODReaderHelpers.cxx
                       src/ASoA.cxx
                       src/AnalysisHelpers.cxx
                       src/AnalysisDataModelHelpers.cxx
                       src/BoostOptionsRetriever.cxx
                       src/ChannelConfigurationPolicy.cxx
                       src/ChannelMatching.cxx
                       src/ChannelConfigurationPolicyHelpers.cxx
                       src/ChannelSpecHelpers.cxx
                       src/CommonDataProcessors.cxx
                       src/CommonServices.cxx
                       src/CommonMessageBackends.cxx
                       src/CompletionPolicy.cxx
                       src/CompletionPolicyHelpers.cxx
                       src/ComputingResourceHelpers.cxx
                       src/ConfigContext.cxx
                       src/ControlService.cxx
                       src/DispatchPolicy.cxx
                       src/ConfigParamStore.cxx
                       src/ConfigParamsHelper.cxx
                       src/DDSConfigHelpers.cxx
                       src/DataAllocator.cxx
                       src/DataDescriptorMatcher.cxx
                       src/DataDescriptorQueryBuilder.cxx
                       src/DataProcessingDevice.cxx
                       src/DataProcessingHeader.cxx
                       src/DataProcessingHelpers.cxx
                       src/SourceInfoHeader.cxx
                       src/DataProcessor.cxx
                       src/DataRelayer.cxx
                       src/DataRelayerHelpers.cxx
                       src/DataSpecUtils.cxx
                       src/DeviceConfigInfo.cxx
                       src/DeviceMetricsInfo.cxx
                       src/DeviceSpec.cxx
                       src/DeviceSpecHelpers.cxx
                       src/DPLMonitoringBackend.cxx
                       src/DriverControl.cxx
                       src/DriverInfo.cxx
                       src/Expressions.cxx
                       src/FairMQDeviceProxy.cxx
                       src/FairMQResizableBuffer.cxx
                       src/FairOptionsRetriever.cxx
                       src/ConfigurationOptionsRetriever.cxx
                       src/FreePortFinder.cxx
                       src/GraphvizHelpers.cxx
                       src/HTTPParser.cxx
                       src/InputRecord.cxx
                       src/InputSpec.cxx
                       src/OutputSpec.cxx
                       src/LifetimeHelpers.cxx
                       src/LocalRootFileService.cxx
                       src/RootConfigParamHelpers.cxx
                       src/LogParsingHelpers.cxx
                       src/MessageContext.cxx
                       src/Metric2DViewIndex.cxx
                       src/SimpleOptionsRetriever.cxx
                       src/O2ControlHelpers.cxx
                       src/OutputSpec.cxx
                       src/PropertyTreeHelpers.cxx
                       src/RCombinedDS.cxx
                       src/ReadoutAdapter.cxx
                       src/ResourcesMonitoringHelper.cxx
                       src/ServiceRegistry.cxx
                       src/SimpleResourceManager.cxx
                       src/SimpleRawDeviceService.cxx
                       src/StreamOperators.cxx
                       src/TMessageSerializer.cxx
                       src/TableBuilder.cxx
                       src/TableConsumer.cxx
                       src/TableTreeHelpers.cxx
                       src/TextDriverClient.cxx
                       src/DataInputDirector.cxx
                       src/DataOutputDirector.cxx
                       src/Task.cxx
                       src/Variant.cxx
                       src/WorkflowCustomizationHelpers.cxx
                       src/WorkflowHelpers.cxx
                       src/WorkflowSerializationHelpers.cxx
                       src/WorkflowSpec.cxx
                       src/WSDriverClient.cxx
                       src/runDataProcessing.cxx
                       src/ExternalFairMQDeviceProxy.cxx
                       src/HistogramRegistry.cxx
                       src/StepTHn.cxx
                       src/SHA1.cxx
                       src/Base64.cxx
                       src/DPLWebSocket.cxx
                       test/TestClasses.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES AliceO2::Common
                                     AliceO2::Configuration
                                     AliceO2::InfoLogger
                                     AliceO2::Monitoring
                                     CURL::libcurl
                                     FairMQ::FairMQ
                                     O2::CommonUtils
                                     O2::MathUtils
                                     O2::FrameworkFoundation
                                     O2::Headers
                                     O2::MemoryResources
                                     O2::PCG
                                     RapidJSON::RapidJSON
                                     arrow::arrow_shared
                                     ms_gsl::ms_gsl
                                     ROOT::ROOTDataFrame
                                     O2::FrameworkLogger
                                     Boost::serialization
                                     ${BOOST_FILESYSTEM}
                                     arrow::gandiva_shared
                                     LibUV::LibUV
                                     )

o2_target_root_dictionary(Framework
                          HEADERS test/TestClasses.h
                          include/Framework/StepTHn.h
                          LINKDEF test/FrameworkCoreTestLinkDef.h)

foreach(t
        AlgorithmSpec
        AnalysisTask
        AnalysisDataModel
        ASoA
        ASoAHelpers
        BoostOptionsRetriever
        ConfigurationOptionsRetriever
        CallbackRegistry
        ChannelSpecHelpers
        CompletionPolicy
        ComputingResourceHelpers
        ConfigParamStore
        ConfigParamRegistry
        DataDescriptorMatcher
        DataProcessorSpec
        DataRefUtils
        DataRelayer
        DeviceConfigInfo
        DeviceMetricsInfo
        DeviceSpec
        DeviceSpecHelpers
        Expressions
        ExternalFairMQDeviceProxy
        FairMQOptionsRetriever
        FairMQResizableBuffer
        FrameworkDataFlowToDDS
        Graphviz
        GroupSlicer
        HistogramRegistry
        HTTPParser
        IndexBuilder
        InfoLogger
        InputRecord
        InputRecordWalker
        InputSpan
        InputSpec
        Kernels
        LogParsingHelpers
        PtrHelpers
        Root2ArrowTable
        RootConfigParamHelpers
        Services
        StringHelpers
        SuppressionGenerator
        TMessageSerializer
        TableBuilder
        TimeParallelPipelining
        TimesliceIndex
        TypeTraits
        Variants
        WorkflowHelpers
        WorkflowSerialization
        TreeToTable
        DataOutputDirector
	DataInputDirector)

  # FIXME ? The NAME parameter of o2_add_test is only needed to help the current
  # o2.sh recipe. If the recipe is changed, those params can go away, if needed.

  o2_add_test(${t} NAME test_Framework_test_${t}
              SOURCES test/test_${t}.cxx
              COMPONENT_NAME Framework
              LABELS framework
              PUBLIC_LINK_LIBRARIES O2::Framework)
endforeach()

o2_add_executable(dpl-null-sink
                  SOURCES src/o2NullSink.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                 )

o2_add_executable(dpl-run
                  SOURCES src/dplRun.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  )

o2_add_executable(verify-aod-file
                  SOURCES src/verifyAODFile.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME Framework)

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
        DataDescriptorMatcher
        DataRelayer
        DeviceMetricsInfo
        InputRecord
        TableBuilder
        WorkflowHelpers
        ASoA
        ASoAHelpers
        HistogramRegistry
        TableToTree
        TreeToTable
        )
  o2_add_executable(benchmark-${b}
                    SOURCES test/benchmark_${b}.cxx
                    COMPONENT_NAME Framework
	            IS_BENCHMARK
                    PUBLIC_LINK_LIBRARIES O2::Framework benchmark::benchmark)
endforeach()

# #####################################################@

foreach(w
        BoostSerializedProcessing
        CallbackService
        RegionInfoCallbackService
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
        SimpleWildcard
        SimpleWildcard02
        SingleDataSource
        Task
        ExternalFairMQDeviceWorkflow
        )
  o2_add_test(${w} NAME test_Framework_test_${w}
              SOURCES test/test_${w}.cxx
              COMPONENT_NAME Framework
              LABELS framework workflow
              PUBLIC_LINK_LIBRARIES O2::Framework
              TIMEOUT 30
              NO_BOOST_TEST
              COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run --shm-segment-size 20000000)
endforeach()

# TODO: DanglingInput test not working for the moment [ERROR] Unable to relay
# part. [WARN] Incoming data is already obsolete, not relaying.
set_property(TEST test_Framework_test_DanglingInputs PROPERTY DISABLED TRUE)

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
    --global-config require-me --run ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS}
    # Note: the group switch makes process consumer parse only the group
    arguments --consumer
    "--global-config consumer-config --local-option hello-aliceo2 --a-boolean3 --an-int2 20 --a-double2 22. --an-int64-2 50000000000000"
  )
