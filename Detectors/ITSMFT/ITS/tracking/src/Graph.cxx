// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Graph.cxx
/// \brief
///

#include "ITStracking/Graph.h"

namespace o2
{
namespace its
{

void Barrier::Wait()
{
  std::unique_lock<std::mutex> lock(mutex);
  if (--count == 0) {
    condition.notify_all();
  } else {
    condition.wait(lock, [this] { return count == 0; });
  }
}

} // namespace its
} // namespace o2