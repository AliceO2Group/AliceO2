// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_UTILITIES_O2DEVICE_COMPATIBILITY_H_
#define ALICEO2_UTILITIES_O2DEVICE_COMPATIBILITY_H_

#include <type_traits>

namespace o2
{
namespace compatibility
{

template <typename T>
class FairMQ13
{
 private:
  template <typename C>
  static std::true_type test(decltype(&C::NewStatePending));
  template <typename C>
  static std::false_type test(...);

 public:
  static bool IsRunning(T* device)
  {
    if constexpr (std::is_same_v<decltype(test<T>(nullptr)), std::true_type>) {
      return !device->NewStatePending();
    } else if constexpr (std::is_same_v<decltype(test<T>(nullptr)), std::false_type>) {
      return device->CheckCurrentState(T::RUNNING);
    }
  }
};

} // namespace compatibility
} // namespace o2

#endif // ALICEO2_UTILITIES_O2DEVICE_COMPATIBILITY_H_
