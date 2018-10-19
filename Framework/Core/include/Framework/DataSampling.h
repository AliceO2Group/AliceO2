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
/// \brief Definition of O2 Data Sampling, v1.0
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <string>

#include "Framework/WorkflowSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DataSamplingPolicy.h"
#include "Framework/ChannelConfigurationPolicy.h"

namespace o2
{
namespace framework
{

/// A class responsible for providing data from main processing flow to QC tasks.
///
/// This class generates message-passing infrastructure to provide desired amount of data to Quality Control tasks or
/// any other clients. Data to be sampled is declared in DataSamplingPolicy'ies configuration file - an example can be
/// found in O2/Framework/TestWorkflows/exampleDataSamplingConfig.json).
///
/// In-code usage:
/// \code{.cxx}
/// void customize(std::vector<CompletionPolicy>& policies)
/// {
///    DataSampling::CustomizeInfrastructure(policies);
/// }
///
/// void customize(std::vector<ChannelConfigurationPolicy>& policies)
/// {
///    DataSampling::CustomizeInfrastructure(policies);
/// }
///
/// #include "Framework/runDataProcessing.h"
///
/// std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext &ctx)
/// {
///   WorkflowSpec workflow;
/// // <declaration of other DPL processors>
///
///   const std::string configurationFilePath = <absolute file path>;
///   DataSampling::GenerateInfrastructure(workflow, configurationFilePath);
///
///   return workflow;
/// }
/// \endcode

//todo: update docu
//todo: clean header mess

class DataSampling
{
 public:
  /// \brief Deleted default constructor. This class is stateless.
  DataSampling() = delete;
  /// \brief Generates data sampling infrastructure.
  /// \param workflow              DPL workflow with already declared data processors which provide data desired by
  ///                              QC tasks.
  /// \param policiesSource        Path to configuration file.
  /// \param threads               Number of dispatcher threads, that will handle the data
  static void GenerateInfrastructure(WorkflowSpec& workflow, const std::string& policiesSource, size_t threads = 1);
  /// \brief Configures dispatcher to consume any data immediately.
  static void CustomizeInfrastructure(std::vector<CompletionPolicy>&);
  /// \brief Applies blocking/nonblocking data sampling configuration to the workflow.
  static void CustomizeInfrastructure(std::vector<ChannelConfigurationPolicy>&);

 private:
  // Internal functions, used by GenerateInfrastructure()
  static std::string dispatcherName();
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATASAMPLING_H
