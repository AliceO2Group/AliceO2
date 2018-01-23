// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_DATASAMPLER_H
#define FRAMEWORK_DATASAMPLER_H

#include <functional>
#include <string>
#include <vector>
#include <random>

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataChunk.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"


namespace o2 {
namespace framework {

class DataSampling {
  public:
    using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

    DataSampling() = delete;

    static void GenerateInfrastructure(WorkflowSpec &workflow, const std::string &configurationSource);

  private:
    struct BernoulliGenerator {
      std::default_random_engine generator;
      std::bernoulli_distribution distribution;

      BernoulliGenerator(double probabilityOfTrue) :
        generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
        distribution(probabilityOfTrue)
      {};
      bool drawLots() {
        return distribution(generator);
      }
    };

    struct QcTaskConfiguration{
      std::string name;
      std::vector<InputSpec> desiredDataSpecs;
      header::DataHeader::SubSpecificationType subSpec;
      double fractionOfDataToSample;
    };
    using QcTaskConfigurations = std::vector<QcTaskConfiguration>;

    struct InfrastructureConfig {
      bool enableTimePipeliningDispatchers;
      bool enableParallelDispatchers;
      bool enableProxy;

      InfrastructureConfig() :
        enableTimePipeliningDispatchers(false),
        enableParallelDispatchers(false),
        enableProxy(false)
      {};
    };

    static void GenerateInfrastructureSimple(WorkflowSpec &workflow, const QcTaskConfigurations &tasks);
    static void GenerateInfrastructureParallel(WorkflowSpec &workflow, const QcTaskConfigurations &tasks);
    static void GenerateInfrastructureTimePipelining(WorkflowSpec &workflow, const QcTaskConfigurations &tasks);

    static AlgorithmSpec::ProcessCallback initCallback(InitContext& ctx);
    static void dispatcherCallback(ProcessingContext &ctx, BernoulliGenerator &bernoulliGenerator);
    static OutputSpec createDispatcherOutputSpec(const InputSpec &dispatcherInput);
    static auto getEdgeMatcher(const SubSpecificationType &subSpec);
    static QcTaskConfigurations readQcTasksConfiguration(const std::string &configurationSource);
    static InfrastructureConfig readInfrastructureConfiguration(const std::string &configurationSource);
};


} //namespace framework
} //namespace o2

#endif //FRAMEWORK_DATASAMPLER_H
