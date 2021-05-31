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

#include "Framework/WorkflowSpec.h"
#include "Framework/InputSpec.h"
#include <string>
#include <boost/property_tree/ptree_fwd.hpp>
#include <optional>

namespace o2::configuration
{
class ConfigurationInterface;
}

namespace o2::framework
{
class CompletionPolicy;
class ChannelConfigurationPolicy;
} // namespace o2::framework

namespace o2::utilities
{

class Dispatcher;

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
  static void GenerateInfrastructure(framework::WorkflowSpec& workflow, const std::string& policiesSource, size_t threads = 1);

  /// \brief Generates data sampling infrastructure.
  /// \param workflow              DPL workflow with already declared data processors which provide data desired by
  ///                              QC tasks.
  /// \param policiesSource        boost::property_tree::ptree with the configuration
  /// \param threads               Number of dispatcher threads, that will handle the data
  static void GenerateInfrastructure(framework::WorkflowSpec& workflow, boost::property_tree::ptree const& policies, size_t threads = 1);
  /// \brief Configures dispatcher to consume any data immediately.
  static void CustomizeInfrastructure(std::vector<framework::CompletionPolicy>&);
  /// \brief Applies blocking/nonblocking data sampling configuration to the workflow.
  static void CustomizeInfrastructure(std::vector<framework::ChannelConfigurationPolicy>&);
  /// \brief Provides InputSpecs to receive data for given DataSamplingPolicy
  static std::vector<framework::InputSpec> InputSpecsForPolicy(const std::string& policiesSource, const std::string& policyName);
  /// \brief Provides InputSpecs to receive data for given DataSamplingPolicy
  /// @deprecated
  static std::vector<framework::InputSpec> InputSpecsForPolicy(configuration::ConfigurationInterface* const config, const std::string& policyName);
  /// \brief Provides InputSpecs to receive data for given DataSamplingPolicy
  static std::vector<framework::InputSpec> InputSpecsForPolicy(std::shared_ptr<configuration::ConfigurationInterface> config, const std::string& policyName);
  /// \brief Provides OutputSpecs of given DataSamplingPolicy
  static std::vector<framework::OutputSpec> OutputSpecsForPolicy(const std::string& policiesSource, const std::string& policyName);
  /// \brief Provides OutputSpecs of given DataSamplingPolicy
  static std::vector<framework::OutputSpec> OutputSpecsForPolicy(configuration::ConfigurationInterface* const config, const std::string& policyName);
  /// \brief Provides the port to be used for a proxy of given DataSamplingPolicy
  static std::optional<uint16_t> PortForPolicy(configuration::ConfigurationInterface* const config, const std::string& policyName);
  /// \brief Provides the port to be used for a proxy of given DataSamplingPolicy
  static std::optional<uint16_t> PortForPolicy(const std::string& policiesSource, const std::string& policyName);
  /// \brief Provides the machines where given DataSamplingPolicy is enabled
  static std::vector<std::string> MachinesForPolicy(configuration::ConfigurationInterface* const config, const std::string& policyName);
  /// \brief Provides the port to be used for a proxy of given DataSamplingPolicy
  static std::vector<std::string> MachinesForPolicy(const std::string& policiesSource, const std::string& policyName);

 private:
  static void DoGenerateInfrastructure(Dispatcher&, framework::WorkflowSpec& workflow, boost::property_tree::ptree const& policies, size_t threads = 1);
  // Internal functions, used by GenerateInfrastructure()
  static std::string createDispatcherName();
};

} // namespace o2::utilities

#endif // FRAMEWORK_DATASAMPLING_H
