// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include <fairmq/FairMQMessage.h>
#include <cassert>

namespace o2 {
namespace framework {

InputRecord::InputRecord(std::vector<InputRoute> const &inputsSchema,
                               std::vector<std::unique_ptr<FairMQMessage>> const& cache)
: mInputsSchema{inputsSchema},
  mCache{cache}
{
  assert(mCache.size() % 2 == 0);
}

int
InputRecord::getPos(const char *binding) const {
  for (int i = 0; i < mInputsSchema.size(); ++i) {
    auto &route = mInputsSchema[i];
    if (route.matcher.binding == binding) {
      return i;
    }
  }
  return -1;
}

int
InputRecord::getPos(std::string const &binding) const {
  for (size_t i = 0; i < mInputsSchema.size(); ++i) {
    auto &route = mInputsSchema[i];
    if (route.matcher.binding == binding) {
      return i;
    }
  }
  return -1;
}

} // namespace framework
} // namespace o2
