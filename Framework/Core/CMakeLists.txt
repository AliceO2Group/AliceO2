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
                       src/ChannelConfigurationPolicyHelpers.cxx
                       src/ChannelSpecHelpers.cxx
                       src/CommonDataProcessors.cxx
                       src/CompletionPolicy.cxx
                       src/CompletionPolicyHelpers.cxx
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
                       src/ExternalFairMQDeviceProxy.cxx
                       src/FairMQDeviceProxy.cxx
                       src/FairMQResizableBuffer.cxx
                       src/FairOptionsRetriever.cxx
                       src/FreePortFinder.cxx
                       src/GraphvizHelpers.cxx
                       src/InputRecord.cxx
                       src/InputSpec.cxx
                       src/LifetimeHelpers.cxx
                       src/LocalRootFileService.cxx
                       src/LogParsingHelpers.cxx
                       src/MessageContext.cxx
                       src/Metric2DViewIndex.cxx
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
                       test/TestClasses.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES AliceO2::Common
                                     AliceO2::Configuration
                                     AliceO2::InfoLogger
                                     AliceO2::Monitoring
                                     CURL::libcurl
                                     FairMQ::FairMQ
                                     FairRoot::Base
                                     O2::CommonUtils
                                     O2::FrameworkFoundation
                                     O2::Headers
                                     O2::MemoryResources
                                     O2::PCG
                                     RapidJSON::RapidJSON
                                     arrow_shared
                                     ms_gsl::ms_gsl
                                     ROOT::ROOTDataFrame
                                     O2::DebugGUI)

o2_target_root_dictionary(Framework
                          HEADERS test/TestClasses.h
                          LINKDEF test/FrameworkCoreTestLinkDef.h)

# O2_GENERATE_EXECUTABLE( EXE_NAME "test_SimpleDataProcessingDevice01" SOURCES
# "test/test_SimpleDataProcessingDevice01.cxx" MODULE_LIBRARY_NAME
# ${LIBRARY_NAME} BUCKET_NAME ${MODULE_BUCKET_NAME} )
#
# O2_GENERATE_EXECUTABLE( EXE_NAME "test_Parallel" SOURCES
# "test/test_Parallel.cxx" MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${MODULE_BUCKET_NAME} )
#
# O2_GENERATE_EXECUTABLE( EXE_NAME "test_TimePipeline" SOURCES
# "test/test_TimePipeline.cxx" MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${MODULE_BUCKET_NAME} )
#
# O2_GENERATE_EXECUTABLE( EXE_NAME "test_DebugGUISokol" SOURCES
# test/test_DebugGUISokol.cxx MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${MODULE_BUCKET_NAME} )
#
# O2_GENERATE_EXECUTABLE( EXE_NAME "test_DebugGUIGL" SOURCES
# test/test_DebugGUIGL.cxx MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${MODULE_BUCKET_NAME} )
#
# Install(FILES test/test_DataSampling.json DESTINATION share/tests/)
#
# set(TEST_SRCS test/test_AnalysisTask.cxx
# test/test_BoostSerializedProcessing.cxx test/test_AlgorithmSpec.cxx
# test/test_BoostOptionsRetriever.cxx test/test_CallbackRegistry.cxx
# test/test_CallbackService.cxx test/test_ChannelSpecHelpers.cxx
# test/test_ContextRegistry.cxx test/test_ConfigParamRegistry.cxx
# test/test_CustomGUISokol.cxx test/test_CustomGUIGL.cxx
# test/test_CompletionPolicy.cxx test/test_DanglingInputs.cxx
# test/test_DanglingOutputs.cxx test/test_DataAllocator.cxx
# test/test_DataDescriptorMatcher.cxx test/test_DataProcessorSpec.cxx
# test/test_DataRefUtils.cxx test/test_DataRelayer.cxx
# test/test_DataSampling.cxx test/test_DataSamplingCondition.cxx
# test/test_DataSamplingPolicy.cxx test/test_DeviceMetricsInfo.cxx
# test/test_DeviceSpec.cxx test/test_ExternalFairMQDeviceProxy.cxx
# test/test_FairMQResizableBuffer.cxx test/test_Forwarding.cxx
# test/test_FrameworkDataFlowToDDS.cxx test/test_FairMQOptionsRetriever.cxx
# test/test_GUITests.cxx test/test_Graphviz.cxx test/test_InfoLogger.cxx
# test/test_InputRecord.cxx test/test_LogParsingHelpers.cxx
# test/test_ParallelProducer.cxx test/test_PtrHelpers.cxx
# test/test_Root2ArrowTable.cxx test/test_Services.cxx
# test/test_SimpleRDataFrameProcessing.cxx
# test/test_SimpleStatefulProcessing01.cxx test/test_SimpleStringProcessing.cxx
# test/test_SingleDataSource.cxx test/test_SimpleTimer.cxx
# test/test_SuppressionGenerator.cxx test/test_TimesliceIndex.cxx
# test/test_TMessageSerializer.cxx test/test_TableBuilder.cxx
# #test/test_Task.cxx test/test_TimeParallelPipelining.cxx
# test/test_TypeTraits.cxx test/test_Variants.cxx test/test_WorkflowHelpers.cxx
# test/test_WorkflowSerialization.cxx test/test_DeviceSpecHelpers.cxx
# test/test_ParallelPipeline.cxx )
#
# set(BENCH_SRCS test/benchmark_ContextRegistry.cxx
# test/benchmark_DataDescriptorMatcher.cxx test/benchmark_DataRelayer.cxx
# test/benchmark_DeviceMetricsInfo.cxx test/benchmark_InputRecord.cxx
# test/benchmark_TableBuilder.cxx )
#
# O2_GENERATE_TESTS( MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# O2FrameworkCore_benchmark_bucket TEST_SRCS ${BENCH_SRCS} )
#
# O2_GENERATE_TESTS( MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${BUCKET_NAME} TEST_SRCS ${TEST_SRCS} TIMEOUT 30 )
#
# # specific tests which needs command line options O2_GENERATE_TESTS(
# MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME ${BUCKET_NAME} TIMEOUT 60
#
# TEST_SRCS test/test_ProcessorOptions.cxx
#
# COMMAND_LINE_ARGS --global-config require-me # Note: the group switch makes
# process consumer parse only the group arguments --consumer "--global-config
# consumer-config --local-option hello-aliceo2 --a-boolean3 --an-int2 20
# --a-double2 22." )
