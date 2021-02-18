// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_CONTEXTREGISTRY_H_
#define O2_FRAMEWORK_CONTEXTREGISTRY_H_

#include "Framework/TypeIdHelpers.h"
#include "Framework/CompilerBuiltins.h"

#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <array>

namespace o2::framework
{

/// @class ContextRegistry
/// Instances are registered by pointer and are not owned by the registry
/// Decouples getting the various contextes from the actual type
/// of context, so that the DataAllocator does not need to know
/// about the various serialization methods.
class ContextRegistry
{
  using ContextElementPtr = void*;
  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static int MAX_CONTEXT_ELEMENTS = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static int MAX_ELEMENTS_MASK = MAX_CONTEXT_ELEMENTS - 1;

 public:
  ContextRegistry();

  template <typename... Types>
  ContextRegistry(Types*... instances)
  {
    set(std::forward<Types*>(instances)...);
  }

  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  T* get() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
    constexpr auto elementId = typeHash & MAX_ELEMENTS_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mElements[i + elementId].first == typeHash) {
        return reinterpret_cast<T*>(mElements[i + elementId].second);
      }
    }
    throw std::runtime_error(std::string("Unable to find context element of kind ") +
                             typeid(T).name() +
                             " did you register one?");
  }

  template <typename T, typename... Types>
  void set(T* instance, Types*... more)
  {
    set(instance);
    set(std::forward<Types*>(more)...);
  }

  // Register a service for the given interface T
  // with actual implementation C, i.e. C is derived from T.
  // Only one instance of type C can be registered per type T.
  // The fact we use a bare pointer indicates that the ownership
  // of the service still belongs to whatever created it, and is
  // not passed to the registry. It's therefore responsibility of
  // the creator of the service to properly dispose it.
  template <typename T>
  void set(T* element)
  {
    static_assert(std::is_void<T>::value == false, "can not register a void object");
    constexpr auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
    constexpr auto elementId = typeHash & MAX_ELEMENTS_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mElements[i + elementId].second == nullptr) {
        mElements[i + elementId].first = typeHash;
        mElements[i + elementId].second = reinterpret_cast<ContextElementPtr>(element);
        return;
      }
    }
    O2_BUILTIN_UNREACHABLE();
  }
 private:
  std::array<std::pair<size_t, ContextElementPtr>, MAX_CONTEXT_ELEMENTS + MAX_DISTANCE> mElements;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTEXTREGISTRY_H_
