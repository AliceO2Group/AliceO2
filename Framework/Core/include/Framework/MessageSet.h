// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_MESSAGESET_H
#define FRAMEWORK_MESSAGESET_H

#include "Framework/PartRef.h"
#include <memory>
#include <vector>

namespace o2
{
namespace framework
{

/// A set of associated inflight messages.
struct MessageSet {
  std::vector<PartRef> parts;

  size_t size() const
  {
    return parts.size();
  }

  void clear()
  {
    parts.clear();
  }

  PartRef& operator[](size_t index)
  {
    return parts[index];
  }

  PartRef const& operator[](size_t index) const
  {
    return parts[index];
  }

  PartRef& at(size_t index)
  {
    return parts.at(index);
  }

  PartRef const& at(size_t index) const
  {
    return parts.at(index);
  }

  decltype(auto) begin()
  {
    return parts.begin();
  }

  decltype(auto) begin() const
  {
    return parts.begin();
  }

  decltype(auto) end()
  {
    return parts.end();
  }

  decltype(auto) end() const
  {
    return parts.end();
  }
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_MESSAGESET_H
