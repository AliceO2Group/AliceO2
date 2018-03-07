// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DISPATCHER_H
#define ALICEO2_DISPATCHER_H

#include <random>

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSamplingConfig.h"

using namespace o2::framework;
using namespace o2::framework::DataSamplingConfig;
using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

namespace o2
{
namespace framework
{

class Dispatcher
{
 public:
  Dispatcher() = delete;
  Dispatcher(const SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task,
             const InfrastructureConfig& cfg)
    : mSubSpec(dispatcherSubSpec),
      mCfg(cfg),
      mDataProcessorSpec()
  {
    mDataProcessorSpec.name = "Dispatcher" + std::to_string(dispatcherSubSpec) + "_for_" + task.name;
  }
  virtual ~Dispatcher() = default;

  DataProcessorSpec getDataProcessorSpec() { return mDataProcessorSpec; };
  SubSpecificationType getSubSpec() { return mSubSpec; };
  virtual void addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                         const std::string& binding) = 0;

 protected:
  /// Bernoulli distribution pseudo-random numbers generator. Used to decide, which data should be bypassed to
  /// QC tasks, in order to achieve certain fraction of data passing through. For example, generator initialized with
  /// value 0.1 returns true *approximately* once per 10 times.
  class BernoulliGenerator
  {
   public:
    BernoulliGenerator(double probabilityOfTrue = 1.0,
                       unsigned int seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count())
      : mGenerator(seed),
        mDistribution(probabilityOfTrue){};
    bool drawLots() { return mDistribution(mGenerator); }

   private:
    std::default_random_engine mGenerator;
    std::bernoulli_distribution mDistribution;
  };

  /// Creates dispatcher output specification basing on input specification of the same data. Basically, it adds '_S' at
  /// the end of description, which makes data stream distinctive from the main flow (which is not sampled/filtered).
  static OutputSpec createDispatcherOutputSpec(const InputSpec& dispatcherInput)
  {
    header::DataDescription description = dispatcherInput.description;
    size_t len = strlen(description.str);
    if (len < description.size - 2) {
      description.str[len] = '_';
      description.str[len + 1] = 'S';
    }

    return OutputSpec{
      dispatcherInput.origin,
      description,
      0,
      static_cast<OutputSpec::Lifetime>(dispatcherInput.lifetime)
    };
  }

 protected:
  SubSpecificationType mSubSpec;
  InfrastructureConfig mCfg;
  DataProcessorSpec mDataProcessorSpec;
};

} // namespace framework
} // namespace o2

#endif // ALICEO2_DISPATCHER_H
