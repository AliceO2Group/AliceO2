// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_DATASAMPLING_H
#define FRAMEWORK_DATASAMPLING_H

/// \file DataSampling.h
/// \brief Definition of O2 Data Sampling, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataChunk.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"

#include "Framework/Dispatcher.h"
#include "Framework/DispatcherDPL.h"
#include "Framework/DispatcherFairMQ.h"
#include "Framework/DispatcherFlpProto.h"
#include "Framework/DataSamplingConfig.h"

namespace o2
{
namespace framework
{

using namespace o2::framework::DataSamplingConfig;

/// A class responsible for providing data from main processing flow to QC tasks.
///
/// This class generates message-passing infrastructure to provide desired amount of data to Quality Control tasks.
/// QC tasks input data should be declared in config file (e.g. O2/Framework/Core/test/exampleDataSamplerConfig.ini ).
/// Data Sampling is based on Data Processing Layer, but supports also standard FairMQ devices by declaring external
/// inputs/outputs in configuration file.
///
/// In-code usage:
/// \code{.cxx}
/// void defineDataProcessing(std::vector<DataProcessorSpec> &workflow)
/// {
///
/// // <declaration of other DPL processors>
///
/// const std::string configurationFilePath = <absolute file path>;
/// DataSampling::GenerateInfrastructure(workflow, configurationFilePath);
///
/// }
/// \endcode

class DataSampling
{
 public:
  /// Deleted default constructor. This class is stateless.
  DataSampling() = delete;
  /// Generates data sampling infrastructure.
  /// \param workflow              DPL workflow with already declared data processors which provide data desired by
  ///                              QC tasks.
  /// \param configurationSource   Path to configuration file.
  static void GenerateInfrastructure(WorkflowSpec& workflow, const std::string& configurationSource);

 private:
  using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

  // Internal functions, used by GenerateInfrastructure()
  static auto getEdgeMatcher(const QcTaskConfiguration& taskCfg);
  static std::unique_ptr<Dispatcher> createDispatcher(SubSpecificationType subSpec, const QcTaskConfiguration& taskCfg,
                                                      InfrastructureConfig infCfg);
  static QcTaskConfigurations readQcTasksConfiguration(const std::string& configurationSource);
  static InfrastructureConfig readInfrastructureConfiguration(const std::string& configurationSource);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATASAMPLING_H
