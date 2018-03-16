// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <memory>

#include "Framework/DataSampling.h"
#include "Framework/runDataProcessing.h"

/// This is an executable allowing to run Data Sampling with FLP Proto, in between FairInjector and FairSampler.
///
/// It also serves as an example of using Data Sampling with both FairMQ inputs and outputs and no processing inside DPL
/// workflow. It is configurable by config file, which is installed at /share/config/dataSamplingFullChainConfig.ini.
/// In source files it is located at O2/Framework/TestWorkflows/dataSamplingFullChainConfig.ini.
/// You can use this file as a template to adapt Data Sampling to your own needs. Input channel configuration is
/// declared in readoutInput/channelConfig, while output channel at readoutQcTaskDefinition/channelConfig. More details
/// regarding options are provided inside config file.
///
/// To run it:
/// > alienv load O2/latest
/// > dataSamplingFullChain
///
/// To obtain full chain (FMQ Readout -> DPL Data Sampling -> FMQ Quality Control) with default configuration:
/// - Readout - set following options in Readout/configDummy.cfg file:
///     \code{.ini}
///     ...
///     exitTimeout=-1
///     ...
///     [sampling]
///     # enable/disable data sampling (1/0)
///     enabled=1
///     # which class of datasampling to use (FairInjector, MockInjector)
///     class=FairInjector
///     ...
///     \endcode
///   And run:
///   \code{}
///   > alienv load Readout/latest
///   > readout.exe file:///your/absolute/path/to/Readout/configDummy.cfg
///   \endcode
/// - Quality Control - configure Quality Control to subscribe for data at port 26525 (or any other, if you change it
///                     in Data Sampling config file). More information on running Quality Control can be found here:
///                     https://github.com/AliceO2Group/QualityControl

using namespace o2::framework;

void defineDataProcessing(std::vector<DataProcessorSpec>& specs)
{
  std::string configFilePath =
    std::string("file://") + getenv("O2_ROOT") + "/share/config/dataSamplingFullChainConfig.ini";
  LOG(INFO) << "Using config file '" << configFilePath << "'";

  DataSampling::GenerateInfrastructure(specs, configFilePath);
}
