// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DATAFLOW_FAKETIMEFRAMEBUILDER_H_
#define DATAFLOW_FAKETIMEFRAMEBUILDER_H_

#include "Headers/DataHeader.h"
#include <vector>
#include <memory>
#include <functional>

namespace o2
{
namespace data_flow
{

struct FakeTimeframeSpec {
  const char* origin;
  const char* dataDescription;
  std::function<void(char*, size_t)> bufferFiller;
  size_t bufferSize;
};

/** Generate a timeframe from the provided specification
  */
std::unique_ptr<char[]> fakeTimeframeGenerator(std::vector<FakeTimeframeSpec>& specs, std::size_t& totalSize);

} // namespace data_flow
} // namespace o2
#endif /* DATAFLOW_FAKETIMEFRAMEBUILDER_H_ */
