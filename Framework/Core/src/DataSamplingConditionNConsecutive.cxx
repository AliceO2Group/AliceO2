// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingConditionNConsecutive.cxx
/// \brief Implementation of DataSamplingConditionNConsecutive
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSamplingCondition.h"
#include "Framework/DataSamplingConditionFactory.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/Logger.h"

namespace o2
{
namespace framework
{

using namespace o2::header;

/// \brief A DataSamplingCondition which approves n consecutive samples in defined cycle. It assumes that timesliceID
// always increments by one.
class DataSamplingConditionNConsecutive : public DataSamplingCondition
{

 public:
  /// \brief Constructor.
  DataSamplingConditionNConsecutive() : DataSamplingCondition(){};
  /// \brief Default destructor
  ~DataSamplingConditionNConsecutive() override = default;

  /// \brief Reads 'samplesNumber' and 'cycleSize'.
  void configure(const boost::property_tree::ptree& config) override
  {
    mSamplesNumber = config.get<size_t>("samplesNumber");
    mCycleSize = config.get<size_t>("cycleSize");
    if (mSamplesNumber > mCycleSize) {
      LOG(WARN) << "Consecutive number of samples is higher than cycle size.";
    }
  };
  /// \brief Makes a positive decision if 'timeslice ID % cycleSize < samplesNumber' is true
  bool decide(const o2::framework::DataRef& dataRef) override
  {
    const auto* dpHeader = get<DataProcessingHeader*>(dataRef.header);
    assert(dpHeader);

    // strongly relying on assumption, that timesliceID always increments by one.
    return dpHeader->startTime % mCycleSize < mSamplesNumber;
  }

 private:
  size_t mSamplesNumber;
  size_t mCycleSize;
};

std::unique_ptr<DataSamplingCondition> DataSamplingConditionFactory::createDataSamplingConditionNConsecutive()
{
  return std::make_unique<DataSamplingConditionNConsecutive>();
}

} // namespace framework
} // namespace o2
