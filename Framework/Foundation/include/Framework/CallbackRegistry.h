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
#ifndef O2_FRAMEWORK_CALLBACKREGISTRY_H_
#define O2_FRAMEWORK_CALLBACKREGISTRY_H_

#include "Framework/RuntimeError.h"
#include "Framework/Pack.h"
#include <functional>

namespace o2::framework
{

template <typename KeyT, KeyT _id, typename CallbackT>
struct RegistryPair {
  using id = std::integral_constant<KeyT, _id>;
  using type = CallbackT;
  type callback;
};

template <typename CallbackId, typename... Args>
class CallbackRegistry
{
 public:
  static constexpr std::size_t size = sizeof...(Args);
  using CallbackTypes = o2::framework::pack<typename Args::type...>;
  using CallbackStore = std::array<std::vector<void*>, size>;
  CallbackRegistry() = default;

  // set callback for slot
  template <CallbackId ID, typename U>
  void set(U&& cb)
  {
    using CallbackType = typename o2::framework::pack_element_t<(int)ID, CallbackTypes>;
    mStore[(int)ID].push_back(reinterpret_cast<void*>(new CallbackType(std::forward<U>(cb))));
  }

  // execute callback at specified slot with argument pack
  template <CallbackId ID, typename... TArgs>
  void call(TArgs&&... args)
  {
    using CallbackType = typename o2::framework::pack_element_t<(int)ID, CallbackTypes>;
    static_assert(std::is_same_v<CallbackType, std::function<void(TArgs...)>>, "callback type mismatch");
    auto& v = mStore[(int)ID];
    for (auto& ptr : v) {
      auto cb = reinterpret_cast<CallbackType*>(ptr);
      if (cb == nullptr) {
        continue;
      }
      (*cb)(std::forward<TArgs>(args)...);
    }
  }

 private:
  // store of RegistryPairs
  CallbackStore mStore;
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_CALLBACKREGISTRY_H_
