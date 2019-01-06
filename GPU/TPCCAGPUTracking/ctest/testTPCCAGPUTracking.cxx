#define BOOST_TEST_MODULE Test TPC CA GPU Tracking
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "AliHLTTPCCAO2Interface.h"
#include "TPCFastTransform.h"
#include "AliGPUCAConfiguration.h"
#include "AliGPUReconstruction.h"

/// @brief Basic test if we can create the interface
BOOST_AUTO_TEST_CASE(CATracking_test1)
{
  auto interface = new AliHLTTPCCAO2Interface;
  
  float solenoidBz = -5.00668; //B-field
  float refX = 1000.; //transport tracks to this x after tracking, >500 for disabling
  bool continuous = true; //time frame data v.s. triggered events

  AliGPUCAConfiguration config;
  config.configProcessing.deviceType = AliGPUReconstruction::DeviceType::CPU;
  config.configProcessing.forceDeviceType = true;
  
  config.configDeviceProcessing.nThreads = 4; //4 threads if we run on the CPU, 1 = default, 0 = auto-detect
  config.configDeviceProcessing.runQA = true; //Run QA after tracking
  config.configDeviceProcessing.eventDisplay = nullptr; //Ptr to event display backend, for running standalone OpenGL event display
  //config.configDeviceProcessing.eventDisplay = new AliGPUCADisplayBackendX11;

  config.configEvent.solenoidBz = solenoidBz;
  config.configEvent.continuousMaxTimeBin = continuous ? 0.023 * 5e6 : 0; //Number of timebins in timeframe if continuous, 0 otherwise

  config.configReconstruction.NWays = 3; //Should always be 3!
  config.configReconstruction.NWaysOuter = true; //Will create outer param for TRD
  config.configReconstruction.SearchWindowDZDR = 2.5f; //Should always be 2.5 for looper-finding and/or continuous tracking
  config.configReconstruction.TrackReferenceX = refX;
  
  interface->Initialize(config, nullptr);
  delete interface;
}
