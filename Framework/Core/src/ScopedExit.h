// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <functional>

namespace o2
{
namespace framework
{

/// Helper class which invokes @a callback when going out of scope.
class ScopedExit
{
 public:
  ScopedExit(std::function<void(void)> callback)
    : mCallback{callback}
  {
  }

  ~ScopedExit()
  {
    mCallback();
  }

 private:
  std::function<void(void)> mCallback;
};

} // namespace framework
} // namespace o2
