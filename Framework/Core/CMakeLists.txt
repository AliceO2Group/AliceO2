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

o2_add_library(Framework
               SOURCES src/AODReaderHelpers.cxx
                       src/ArrowSupport.cxx
                       src/ArrowTableSlicingCache.cxx
                       src/AnalysisDataModel.cxx
                       src/ASoA.cxx
                       src/AsyncQueue.cxx
                       src/AnalysisDataModelHelpers.cxx
                       src/BoostOptionsRetriever.cxx
                       src/CallbacksPolicy.cxx
                       src/ChannelConfigurationPolicy.cxx
                       src/ChannelMatching.cxx
                       src/ChannelConfigurationPolicyHelpers.cxx
                       src/ChannelSpecHelpers.cxx
                       src/CCDBParamSpec.cxx
                       src/CommandInfo.cxx
                       src/CommonDataProcessors.cxx
                       src/CommonServices.cxx
                       src/CommonMessageBackends.cxx
                       src/CommonDriverServices.cxx
                       src/ControlWebSocketHandler.cxx
                       src/CompletionPolicy.cxx
                       src/CompletionPolicyHelpers.cxx
                       src/ComputingQuotaEvaluator.cxx
                       src/ComputingResourceHelpers.cxx
                       src/ConfigContext.cxx
                       src/ControlService.cxx
                       src/ControlServiceHelpers.cxx
                       src/ControlWebSocketHandler.cxx
                       src/DispatchPolicy.cxx
                       src/DataSender.cxx
                       src/DataProcessingStats.cxx
                       src/DataProcessingStates.cxx
                       src/DefaultsHelpers.cxx
                       src/DomainInfoHeader.cxx
                       src/ProcessingPoliciesHelpers.cxx
                       src/ConfigParamStore.cxx
                       src/ConfigParamsHelper.cxx
                       src/ChannelParamSpec.cxx
                       src/DDSConfigHelpers.cxx
                       src/DataAllocator.cxx
                       src/DataDescriptorMatcher.cxx
                       src/DataDescriptorQueryBuilder.cxx
                       src/DataProcessingDevice.cxx
                       src/DataProcessingHeader.cxx
                       src/DataProcessingHelpers.cxx
                       src/DataProcessorSpecHelpers.cxx
                       src/DataProcessorMatchers.cxx
                       src/DataProcessingContext.cxx
                       src/DataRefUtils.cxx
                       src/SourceInfoHeader.cxx
                       src/DataProcessor.cxx
                       src/DataRelayer.cxx
                       src/DataRelayerHelpers.cxx
                       src/DataSpecUtils.cxx
                       src/DeviceConfigInfo.cxx
                       src/DevicesManager.cxx
                       src/DeviceMetricsInfo.cxx
                       src/DeviceMetricsHelper.cxx
                       src/DeviceSpec.cxx
                       src/DeviceController.cxx
                       src/DeviceSpecHelpers.cxx
                       src/DeviceStateHelpers.cxx
                       src/DPLMonitoringBackend.cxx
                       src/DriverControl.cxx
                       src/DriverClient.cxx
                       src/DriverInfo.cxx
                       src/Expressions.cxx
                       src/FairMQDeviceProxy.cxx
                       src/FairMQResizableBuffer.cxx
                       src/FairOptionsRetriever.cxx
                       src/ConfigurationOptionsRetriever.cxx
                       src/FreePortFinder.cxx
                       src/GraphvizHelpers.cxx
                       src/MermaidHelpers.cxx
                       src/HTTPParser.cxx
                       src/IndexBuilderHelpers.cxx
                       src/InputRecord.cxx
                       src/InputRouteHelpers.cxx
                       src/InputSpan.cxx
                       src/InputSpec.cxx
                       src/OutputSpec.cxx
                       src/LifetimeHelpers.cxx
                       src/LocalRootFileService.cxx
                       src/RootConfigParamHelpers.cxx
                       src/StringContext.cxx
                       src/LogParsingHelpers.cxx
                       src/MessageContext.cxx
                       src/Metric2DViewIndex.cxx
                       src/SimpleOptionsRetriever.cxx
                       src/O2ControlHelpers.cxx
                       src/O2ControlLabels.cxx
                       src/O2ControlParameters.cxx
                       src/O2DataModelHelpers.cxx
                       src/OutputSpec.cxx
                       src/OptionsHelpers.cxx
                       src/PropertyTreeHelpers.cxx
                       src/ProcessingContext.cxx
                       src/Plugins.cxx
                       src/RateLimiter.cxx
                       src/ResourcesMonitoringHelper.cxx
                       src/ResourcePolicy.cxx
                       src/ResourcePolicyHelpers.cxx
                       src/SendingPolicy.cxx
                       src/ServiceRegistry.cxx
                       src/ServiceSpec.cxx
                       src/SimpleResourceManager.cxx
                       src/SimpleRawDeviceService.cxx
                       src/StreamOperators.cxx
                       src/StreamContext.cxx
                       src/TMessageSerializer.cxx
                       src/TableBuilder.cxx
                       src/TableConsumer.cxx
                       src/TableTreeHelpers.cxx
                       src/TopologyPolicy.cxx
                       src/TextDriverClient.cxx
                       src/TimesliceIndex.cxx
                       src/TimingHelpers.cxx
                       src/DataOutputDirector.cxx
                       src/Task.cxx
                       src/Array2D.cxx
                       src/Variant.cxx
                       src/WorkflowCustomizationHelpers.cxx
                       src/WorkflowHelpers.cxx
                       src/WorkflowSerializationHelpers.cxx
                       src/WorkflowSpec.cxx
                       src/WSDriverClient.cxx
                       src/runDataProcessing.cxx
                       src/ExternalFairMQDeviceProxy.cxx
                       src/HistogramSpec.cxx
                       src/HistogramRegistry.cxx
                       src/StepTHn.cxx
                       src/Base64.cxx
                       src/DPLWebSocket.cxx
                       src/TimerParamSpec.cxx
                       test/TestClasses.cxx
               TARGETVARNAME targetName
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES AliceO2::Configuration
                                     AliceO2::Monitoring
                                     CURL::libcurl
                                     FairMQ::FairMQ
                                     ROOT::Tree
                                     ROOT::Hist
                                     O2::FrameworkFoundation
                                     O2::CommonConstants
                                     O2::Headers
                                     O2::MemoryResources
                                     O2::PCG
                                     RapidJSON::RapidJSON
                                     Arrow::arrow_shared
                                     Microsoft.GSL::GSL
                                     O2::FrameworkLogger
                                     Gandiva::gandiva_shared
                                     LibUV::LibUV
                                     )

# To get the necessary include for the MC status codes. Needs to be public, for instance O2Physics heavily depends on Framework
target_include_directories(${targetName} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/DataFormats/simulation/include>)

o2_target_root_dictionary(Framework
                          HEADERS test/TestClasses.h
                          include/Framework/StepTHn.h
                          LINKDEF test/FrameworkCoreTestLinkDef.h)

add_executable(o2-test-framework-core
              test/test_AlgorithmSpec.cxx
              test/test_AnalysisTask.cxx
              test/test_AnalysisDataModel.cxx
              test/test_AsyncQueue.cxx
              test/test_ASoA.cxx
              test/test_ASoAHelpers.cxx
              test/test_BoostOptionsRetriever.cxx
              test/test_ConfigurationOptionsRetriever.cxx
              test/test_ChannelSpecHelpers.cxx
              test/test_CheckTypes.cxx
              test/test_CompletionPolicy.cxx
              test/test_ComputingResourceHelpers.cxx
              test/test_ComputingQuotaEvaluator.cxx
              test/test_ControlServiceHelpers.cxx
              test/test_ConfigParamStore.cxx
              test/test_ConfigParamRegistry.cxx
              test/test_DataDescriptorMatcher.cxx
              test/test_DataDescriptorQueryBuilder.cxx
              test/test_DataProcessingStats.cxx
              test/test_DataProcessingStates.cxx
              test/test_DataProcessorSpec.cxx
              test/test_DataRefUtils.cxx
              test/test_DataRelayer.cxx
              test/test_DeviceConfigInfo.cxx
              test/test_DeviceMetricsInfo.cxx
              test/test_DeviceSpec.cxx
              test/test_DeviceSpecHelpers.cxx
              test/test_DeviceStateHelpers.cxx
              test/test_Expressions.cxx
              test/test_ExternalFairMQDeviceProxy.cxx
              test/test_FairMQOptionsRetriever.cxx
              test/test_FairMQResizableBuffer.cxx
              test/test_FairMQ.cxx
              test/test_FrameworkDataFlowToDDS.cxx
              test/test_FrameworkDataFlowToO2Control.cxx
              test/test_Graphviz.cxx
              test/test_GroupSlicer.cxx
              test/test_HistogramRegistry.cxx
              test/test_HTTPParser.cxx
              test/test_IndexBuilder.cxx
              test/test_InputRecord.cxx
              test/test_InputRecordWalker.cxx
              test/test_InputSpan.cxx
              test/test_InputSpec.cxx
              test/test_LogParsingHelpers.cxx
              test/test_Mermaid.cxx
              test/test_OptionsHelpers.cxx
              test/test_OverrideLabels.cxx
              test/test_O2DataModelHelpers.cxx
              test/test_PtrHelpers.cxx
              test/test_RootConfigParamHelpers.cxx
              test/test_Services.cxx
              test/test_StringHelpers.cxx
              test/test_StaticFor.cxx
              test/test_TMessageSerializer.cxx
              test/test_TableBuilder.cxx
              test/test_TimeParallelPipelining.cxx
              test/test_TimesliceIndex.cxx
              test/test_TypeTraits.cxx
              test/test_Variants.cxx
              test/test_WorkflowHelpers.cxx
              test/test_WorkflowSerialization.cxx
              test/test_TreeToTable.cxx
              test/test_DataOutputDirector.cxx
              test/unittest_SimpleOptionsRetriever.cxx
              test/unittest_DataSpecUtils.cxx
            )
target_link_libraries(o2-test-framework-core PRIVATE O2::Framework)
target_link_libraries(o2-test-framework-core PRIVATE O2::Catch2)

get_filename_component(outdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../tests ABSOLUTE)
set_property(TARGET o2-test-framework-core PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})

add_test(NAME framework:core COMMAND o2-test-framework-core --skip-benchmarks)

o2_add_test(AlgorithmWrapper NAME test_Framework_test_AlgorithmWrapper
            SOURCES test/test_AlgorithmWrapper.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework)

o2_add_test(Timers NAME test_Framework_test_Timers
            SOURCES test/test_Timers.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework)

o2_add_test(SuppressionGenerator NAME test_Framework_test_SuppressionGenerator
            SOURCES test/test_SuppressionGenerator.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework)

o2_add_test(O2DatabasePDG NAME test_Framework_test_O2DatabasePDG
            SOURCES test/test_O2DatabasePDG.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework O2::FrameworkPhysicsSupport)

# All the tests which require ROOT to work
add_executable(o2-test-framework-root
               test/test_Root2ArrowTable.cxx
               )
target_link_libraries(o2-test-framework-root PRIVATE O2::Framework)
target_link_libraries(o2-test-framework-root PRIVATE O2::Catch2)
target_link_libraries(o2-test-framework-root PRIVATE ROOT::ROOTDataFrame)
set_property(TARGET o2-test-framework-root PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})
add_test(NAME framework:root COMMAND o2-test-framework-root --skip-benchmarks)

o2_add_test(InfoLogger NAME test_Framework_test_InfoLogger
            SOURCES test/test_InfoLogger.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework AliceO2::InfoLogger)

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
                  PUBLIC_LINK_LIBRARIES O2::Framework ROOT::TreePlayer
                  COMPONENT_NAME Framework)

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
        EventMixing
        HistogramRegistry
        TableToTree
        TreeToTable
        ExternalFairMQDeviceProxies
        )
  o2_add_executable(benchmark-${b}
                    SOURCES test/benchmark_${b}.cxx
                    COMPONENT_NAME Framework
                    IS_BENCHMARK
                    PUBLIC_LINK_LIBRARIES O2::Framework benchmark::benchmark)
endforeach()

# #####################################################@

foreach(w
        CallbackService
        RegionInfoCallbackService
        DanglingInputs
        DanglingOutputs
        DataAllocator
        StaggeringWorkflow
        Forwarding
        ParallelPipeline
        ParallelProducer
        SlowConsumer
        SlowProducerWithWildCard
        SimpleDataProcessingDevice01
        SimpleStatefulProcessing01
        SimpleStringProcessing
        SimpleTimer
        SimpleWildcard
        SimpleWildcard02
        SingleDataSource
        Task
        ExternalFairMQDeviceWorkflow
        VariablePayloadSequenceWorkflow
        DataDescriptorMatcherWorkflow
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

if (BUILD_TESTING)
# TODO: DanglingInput test not working for the moment [ERROR] Unable to relay
# part. [WARN] Incoming data is already obsolete, not relaying.
set_property(TEST test_Framework_test_DanglingInputs PROPERTY DISABLED TRUE)

endif()

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
    # Note: with group switch, process 'consumer' will only parse the group arguments
    --consumer
    "--global-config consumer-config --local-option hello-aliceo2 --a-boolean3 --an-int2 20 --a-double2 22. --an-int64-2 50000000000000"
  )

# the test is compiled from the ExternalFairMQDeviceWorkflow test and run with
# command line option to include the output proxy
o2_add_test(
  ExternalFairMQOutputProxyWorkflow NAME test_Framework_test_ExternalFairMQOutputProxyWorkflow
  SOURCES test/test_ExternalFairMQDeviceWorkflow.cxx
  COMPONENT_NAME Framework
  LABELS framework workflow
  TIMEOUT 60
  PUBLIC_LINK_LIBRARIES O2::Framework
  NO_BOOST_TEST
  COMMAND_LINE_ARGS
    --proxy-mode all --run ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS}
  )
