// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_CONTEXTREGISTRY_H
#define FRAMEWORK_CONTEXTREGISTRY_H

#include "Framework/TypeIdHelpers.h"

#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <array>

namespace o2
{
namespace framework
{

/// @class ContextRegistry
/// Instances are registered by pointer and are not owned by the registry
/// Decouples getting the various contextes from the actual type
/// of context, so that the DataAllocator does not need to know
/// about the various serialization methods.
///
class ContextRegistry
{
 public:
  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static int MAX_CONTEXT = 32;
  /// The mask to use to calculate the initial slot id.
  constexpr static int MAX_CONTEXT_MASK = MAX_CONTEXT - 1;

  ContextRegistry();

  template <typename... Types>
  ContextRegistry(Types*... instances)
  {
    mRegistryKey.fill(0);
    mRegistryValue.fill(nullptr);
    set(std::forward<Types*>(instances)...);
  }

  template <typename T>
  T* get() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
    return reinterpret_cast<T*>(get(typeHash));
  }

  template <typename T, typename... Types>
  void set(T* instance, Types*... more)
  {
    set(instance);
    set(std::forward<Types*>(more)...);
  }

  template <typename T>
  void set(T* instance)
  {
    static_assert(std::is_void<T>::value == false, "can not register a void object");
    auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
    set(reinterpret_cast<void*>(instance), typeHash);
  }

  void* get(uint32_t typeHash) const
  {
    auto id = typeHash & MAX_CONTEXT_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mRegistryKey[i + id] == typeHash) {
        return mRegistryValue[i + id];
      }
    }
    return nullptr;
  }

  void set(void* instance, uint32_t typeHash)
  {
    auto id = typeHash & MAX_CONTEXT_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mRegistryValue[i + id] == nullptr) {
        mRegistryKey[i + id] = typeHash;
        mRegistryValue[i + id] = instance;
        return;
      }
    }
  }

 private:
  std::array<uint32_t, MAX_CONTEXT + MAX_DISTANCE> mRegistryKey;
  std::array<void*, MAX_CONTEXT + MAX_DISTANCE> mRegistryValue;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CONTEXTREGISTRY_H
