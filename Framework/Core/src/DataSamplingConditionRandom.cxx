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
#include "Framework/DataSamplingConditionFactory.h"
#include "Framework/DataProcessingHeader.h"

#include "PCG/pcg_random.hpp"

namespace o2
{
namespace framework
{

// todo: consider using run number as a seed
using namespace o2::header;

/// \brief A DataSamplingCondition which makes decisions randomly, but with determinism.
class DataSamplingConditionRandom : public DataSamplingCondition
{

 public:
  /// \brief Constructor.
  DataSamplingConditionRandom() : DataSamplingCondition(),
                                  mThreshold(0),
                                  mGenerator(0),
                                  mCurrentTimesliceID(0),
                                  mLastDecision(false){};
  /// \brief Default destructor
  ~DataSamplingConditionRandom() override = default;

  /// \brief Reads 'fraction' parameter (type double, between 0 and 1) and seed (int).
  void configure(const boost::property_tree::ptree& config) override
  {
    mThreshold = static_cast<uint32_t>(config.get<double>("fraction") * std::numeric_limits<uint32_t>::max());
    mGenerator.seed(config.get<uint64_t>("seed"));
    mCurrentTimesliceID = 0;
    mLastDecision = false;
  };
  /// \brief Makes pseudo-random, deterministic decision based on TimesliceID.
  /// The reason behind using TimesliceID is to ensure, that data of the same events is sampled even on different FLPs.
  bool decide(const o2::framework::DataRef& dataRef) override
  {
    const auto* dpHeader = get<DataProcessingHeader*>(dataRef.header);
    assert(dpHeader);

    int64_t diff = dpHeader->startTime - mCurrentTimesliceID;
    if (diff == -1) {
      return mLastDecision;
    } else if (diff < -1) {
      mGenerator.backstep(static_cast<uint64_t>(-diff));
    } else if (diff > 0) {
      mGenerator.advance(static_cast<uint64_t>(diff));
    }

    mLastDecision = mGenerator() < mThreshold;
    mCurrentTimesliceID = dpHeader->startTime + 1;
    return mLastDecision;
  }

 private:
  uint32_t mThreshold;
  pcg32_fast mGenerator;
  bool mLastDecision;
  DataProcessingHeader::StartTime mCurrentTimesliceID;
};

std::unique_ptr<DataSamplingCondition> DataSamplingConditionFactory::createDataSamplingConditionRandom()
{
  return std::make_unique<DataSamplingConditionRandom>();
}

} // namespace framework
} // namespace o2
