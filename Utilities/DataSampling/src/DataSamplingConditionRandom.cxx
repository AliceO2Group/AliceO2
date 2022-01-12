// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingConditionRandom.cxx
/// \brief Implementation of random DataSamplingCondition
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "DataSampling/DataSamplingCondition.h"
#include "DataSampling/DataSamplingConditionFactory.h"
#include "Headers/DataHeader.h"
#include "Framework/DataProcessingHeader.h"

#include "PCG/pcg_random.hpp"
#include <random>

#include <boost/property_tree/ptree.hpp>

using namespace o2::framework;

namespace o2::utilities
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

    auto seed = config.get<uint64_t>("seed");
    mGenerator.seed((seed == 0) ? std::random_device()() : seed);

    mCurrentTimesliceID = 0;
    mLastDecision = false;

    auto timeslideID = config.get_optional<std::string>("timesliceId").value_or("startTime");
    if (timeslideID == "startTime") {
      mGetTimesliceID = [](const o2::framework::DataRef& dataRef) {
        const auto* dph = get<DataProcessingHeader*>(dataRef.header);
        assert(dph);
        return dph->startTime;
      };
    } else if (timeslideID == "tfCounter") {
      mGetTimesliceID = [](const o2::framework::DataRef& dataRef) {
        const auto* dh = get<DataHeader*>(dataRef.header);
        assert(dh);
        return dh->tfCounter;
      };
    } else if (timeslideID == "firstTForbit") {
      mGetTimesliceID = [](const o2::framework::DataRef& dataRef) {
        const auto* dh = get<DataHeader*>(dataRef.header);
        assert(dh);
        return dh->firstTForbit;
      };
    } else {
      throw std::runtime_error("Data Sampling Condition Random does not support timesliceId '" + timeslideID + "'");
    }
  };
  /// \brief Makes pseudo-random, deterministic decision based on TimesliceID.
  /// The reason behind using TimesliceID is to ensure, that data of the same events is sampled even on different FLPs.
  bool decide(const o2::framework::DataRef& dataRef) override
  {
    auto tid = mGetTimesliceID(dataRef);

    int64_t diff = tid - mCurrentTimesliceID;
    if (diff == -1) {
      return mLastDecision;
    } else if (diff < -1) {
      mGenerator.backstep(static_cast<uint64_t>(-diff));
    } else if (diff > 0) {
      mGenerator.advance(static_cast<uint64_t>(diff));
    }

    mLastDecision = mGenerator() < mThreshold;
    mCurrentTimesliceID = tid + 1;
    return mLastDecision;
  }

 private:
  uint32_t mThreshold;
  pcg32_fast mGenerator;
  bool mLastDecision;
  uint64_t mCurrentTimesliceID;
  std::function<uint64_t(const o2::framework::DataRef&)> mGetTimesliceID;
};

std::unique_ptr<DataSamplingCondition> DataSamplingConditionFactory::createDataSamplingConditionRandom()
{
  return std::make_unique<DataSamplingConditionRandom>();
}

} // namespace o2::utilities
