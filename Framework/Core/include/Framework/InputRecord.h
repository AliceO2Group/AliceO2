// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTRECORD_H
#define FRAMEWORK_INPUTRECORD_H

#include "Framework/DataRef.h"
#include "Framework/InputRoute.h"

#include <fairmq/FairMQMessage.h>

#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <exception>
#include <memory>

namespace o2 {
namespace framework {

struct InputSpec;

/// This class holds the inputs which  are being processed by the system while
/// they are  being processed.  The user can  get an instance  for it  via the
/// ProcessingContext and can use it to retrieve the inputs, either by name or
/// by index.  A few utility  methods are  provided to automatically  cast the
/// inputs to  known types. The user  is also allowed to  override the `getAs`
/// template and provide his own serialization mechanism.
class InputRecord {
public:
  InputRecord(std::vector<InputRoute> const &inputs,
                 std::vector<std::unique_ptr<FairMQMessage>> const &cache);

  int getPos(const char *name) const;
  int getPos(const std::string &name) const;

  DataRef getByPos(int pos) const {
    if (pos*2 >= mCache.size() || pos < 0) {
      throw std::runtime_error("Unknown argument requested");
    }
    assert(pos >= 0);
    return DataRef{&mInputsSchema[pos].matcher,
                   static_cast<char const*>(mCache[pos*2]->GetData()),
                   static_cast<char const*>(mCache[pos*2+1]->GetData())};
  }

  DataRef get(const char *name) const {
    return getByPos(getPos(name));
  }

  DataRef get(std::string const &name) const {
    return getByPos(getPos(name));
  }

  template <class T>
  T const &getAs(char const *name) const {
    return *reinterpret_cast<T const *>(get(name).payload);
  }

  size_t size() {
    return mCache.size()/2;
  }

private:
  std::vector<InputRoute> const &mInputsSchema;
  std::vector<std::unique_ptr<FairMQMessage>> const &mCache;
};

} // framework
} // o2

#endif // FRAMEWORK_INPUTREGISTRY_H
