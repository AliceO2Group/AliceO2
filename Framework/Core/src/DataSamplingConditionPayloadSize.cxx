// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingConditionPayloadSize.cxx
/// \brief Implementation of DataSamplingConditionPayloadSize
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSamplingCondition.h"
#include "Framework/DataSamplingConditionFactory.h"
#include "Headers/DataHeader.h"
#include "Framework/Logger.h"

namespace o2
{
namespace framework
{

using namespace o2::header;

/// \brief A DataSamplingCondition which makes decisions based on payload size.
class DataSamplingConditionPayloadSize : public DataSamplingCondition
{

 public:
  /// \brief Constructor.
  DataSamplingConditionPayloadSize() : DataSamplingCondition(), mLowerLimit(1), mUpperLimit(0){};
  /// \brief Default destructor
  ~DataSamplingConditionPayloadSize() override = default;

  /// \brief Reads 'lowerLimit' and 'UpperLimit' of allowed payload size.
  void configure(const boost::property_tree::ptree& config) override
  {
    mLowerLimit = config.get<size_t>("lowerLimit");
    mUpperLimit = config.get<size_t>("upperLimit");
    if (mLowerLimit > mUpperLimit) {
      LOG(WARN) << "Lower limit is higher than upper limit.";
    }
  };
  /// \brief Makes a positive decision if the payload size is within given limits
  bool decide(const o2::framework::DataRef& dataRef) override
  {
    const auto* header = get<DataHeader*>(dataRef.header);
    assert(header);

    return header->payloadSize >= mLowerLimit && header->payloadSize <= mUpperLimit;
  }

 private:
  size_t mLowerLimit;
  size_t mUpperLimit;
};

std::unique_ptr<DataSamplingCondition> DataSamplingConditionFactory::createDataSamplingConditionPayloadSize()
{
  return std::make_unique<DataSamplingConditionPayloadSize>();
}

} // namespace framework
} // namespace o2
