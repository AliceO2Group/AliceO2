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

/// \file DataSampling.h
/// \brief Definition of O2 Data Sampling, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <functional>
#include <string>
#include <vector>
#include <random>

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataChunk.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include <Framework/SimpleRawDeviceService.h>
#include "FairMQDevice.h"
#include "FairMQTransportFactory.h"

namespace o2 {
namespace framework {


/// A class responsible for providing data from main processing flow to QC tasks.
///
/// This class generates message-passing infrastructure to provide desired amount of data to Quality Control tasks.
/// QC tasks input data should be declared in config file (e.g. O2/Framework/Core/test/exampleDataSamplerConfig.ini ).
/// Data Sampling is based on Data Processing Layer, but supports also standard FairMQ devices by declaring external
/// inputs/outputs in configuration file.
///
/// In-code usage:
/// void defineDataProcessing(std::vector<DataProcessorSpec> &workflow)
/// {
///
/// // <declaration of other DPL processors>
///
/// std::string configurationFilePath = <file path>;
/// DataSampling::GenerateInfrastructure(workflow, configurationFilePath);
///
/// }

class DataSampling {
  public:

    /// Deleted default constructor. This class is stateless.
    DataSampling() = delete;

    /// Generates data sampling infrastructure.
    /// \param workflow              DPL workflow with already declared data processors which provide data desired by
    ///                              QC tasks.
    /// \param configurationSource   Path to configuration file.
    static void GenerateInfrastructure(WorkflowSpec &workflow, const std::string &configurationSource);

  private:
    using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

    /// Bernoulli distribution pseudo-random numbers generator. Used to decide, which data should be bypassed to
    /// QC tasks, in order to achieve certain fraction of data passing through. For example, generator initialized with
    /// value 0.1 returns true *approximately* once per 10 times.
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

    /// Structure that holds requirements for external FairMQ data. Probably temporary.
    struct FairMqInput {
      OutputSpec outputSpec;
      std::string channelConfig;
      std::string converterType;
    };

    /// Structure that holds QC task requirements for sampled data.
    struct QcTaskConfiguration{
      std::string name;
      std::vector<FairMqInput> desiredFairMqData; //for temporary feature
      std::vector<InputSpec> desiredDataSpecs;
      SubSpecificationType subSpec;
      double fractionOfDataToSample;
      std::string fairMqOutputChannelConfig;
    };
    using QcTaskConfigurations = std::vector<QcTaskConfiguration>;

    /// Structure that holds general data sampling infrastructure configuration
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

    // Callbacks of Data Processors to be invoked by Data Processing Layer
    /// Dispatcher initialization callback
    static AlgorithmSpec::ProcessCallback dispatcherInitCallback(InitContext &ctx);
    /// Main dispatcher callback with DPL outputs
    static void dispatcherCallback(ProcessingContext &ctx, BernoulliGenerator &bernoulliGenerator);

    /// Dispatcher with FairMQ output initialization callback
    static AlgorithmSpec::ProcessCallback dispatcherInitCallbackFairMQ(InitContext &ctx, const std::string &channel,
                                                                       double fraction);
    /// Main dispatcher callback with FairMQ output
    static void dispatcherCallbackFairMQ(ProcessingContext &ctx, BernoulliGenerator &bernoulliGenerator,
                                         FairMQDevice* device, const std::string &channel);

    // Other internal functions, used by GenerateInfrastructure()
    static OutputSpec createDispatcherOutputSpec(const InputSpec &dispatcherInput);
    static auto getEdgeMatcher(const QcTaskConfiguration &taskCfg);
    static auto getDispatcherCreator(const QcTaskConfiguration & taskCfg);
    static auto getEdgeCreator(const QcTaskConfiguration & taskCfg, const InfrastructureConfig &infrastructureCfg);
    static QcTaskConfigurations readQcTasksConfiguration(const std::string &configurationSource);
    static InfrastructureConfig readInfrastructureConfiguration(const std::string &configurationSource);
};


} //namespace framework
} //namespace o2

#endif //FRAMEWORK_DATASAMPLER_H
