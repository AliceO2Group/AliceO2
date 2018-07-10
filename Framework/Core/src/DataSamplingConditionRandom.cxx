// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingConditionRandom.cxx
/// \brief Implementation of random DataSamplingCondition
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSamplingCondition.h"
#include "Framework/DataProcessingHeader.h"

#include <TRandom3.h>

namespace o2
{
namespace framework
{

// todo: choose the best PRNG (TRandom3 is fast, but i am not sure about its statistical soundness and behaviour with
// very small percents).
// todo: consider using run number as a seed

using namespace o2::header;

/// \brief A DataSamplingCondition which makes decicions randomly, but with determinism.
class DataSamplingConditionRandom : public DataSamplingCondition
{

 public:
  /// \brief Constructor.
  DataSamplingConditionRandom() : DataSamplingCondition(), mSeed(0), mFraction(0.0), mGenerator(0){};
  /// \brief Default destructor
  ~DataSamplingConditionRandom() = default;

  /// \brief Reads 'fraction' parameter (type double, between 0 and 1) and seed (int).
  void configure(const o2::configuration::tree::Branch& cfg) override
  {
    mFraction = configuration::tree::getRequired<double>(cfg, "fraction");
    mSeed = configuration::tree::getRequired<int>(cfg, "seed");
  };
  /// \brief Makes pseudo-random, deterministic decision based on TimesliceID.
  /// The reason behind using TimesliceID is to ensure, that data of the same events is sampled even on different FLPs.
  bool decide(const o2::framework::DataRef& dataRef) override
  {
    const auto* dpHeader = get<DataProcessingHeader*>(dataRef.header);
    assert(dpHeader);

    mGenerator.SetSeed(dpHeader->startTime + mSeed);
    return static_cast<bool>(mGenerator.Binomial(1, mFraction));
  }

 private:
  int mSeed;
  double mFraction;
  TRandom3 mGenerator;
};

std::unique_ptr<DataSamplingCondition> DataSamplingCondition::getDataSamplingConditionRandom()
{
  return std::make_unique<DataSamplingConditionRandom>();
}

} // namespace framework
} // namespace o2