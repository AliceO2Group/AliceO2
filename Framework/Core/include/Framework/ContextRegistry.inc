// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/TypeIdHelpers.h"
#include "Framework/CompilerBuiltins.h"
#include <stdexcept>
#include <string>


namespace o2::framework {

namespace detail {
template<class Lambda, int=(Lambda{}(), 0)>
static constexpr bool is_constexpr(Lambda) { return true; }
static constexpr bool is_constexpr(...) { return false; }
}

template <typename... Types>
inline ContextRegistry::ContextRegistry(Types*... instances)
{
  set(std::forward<Types*>(instances)...);
}

/// Get a service for the given interface T. The returned reference exposed to
/// the user is actually of the last concrete type C registered, however this
/// should not be a problem.
template <typename T>
inline T* ContextRegistry::get() const
{
  constexpr auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
  int h = 0;
  if constexpr (detail::is_constexpr([]() { return __PRETTY_FUNCTION__; })) {
    h = typeHash;
  } else {
    h = typeid(T*).hash_code();
  }
  auto elementId = h & MAX_ELEMENTS_MASK;

  for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
    if (mElements[i + elementId].first == h) {
      return reinterpret_cast<T*>(mElements[i + elementId].second);
    }
  }
  throw std::runtime_error(std::string("Unable to find context element of kind ") +
                           typeid(T).name() +
                           " did you register one?");
}

template <typename T, typename... Types>
inline void ContextRegistry::set(T* instance, Types*... more)
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
inline void ContextRegistry::set(T* element)
{
  static_assert(std::is_void<T>::value == false, "can not register a void object");
  constexpr auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
  int h = 0;
  if constexpr (detail::is_constexpr([]() { return __PRETTY_FUNCTION__; })) {
    h = typeHash;
  } else {
    h = typeid(T*).hash_code();
  }
  auto elementId = h & MAX_ELEMENTS_MASK;
  for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
    if (mElements[i + elementId].second == nullptr) {
      mElements[i + elementId].first = h;
      mElements[i + elementId].second = reinterpret_cast<ContextElementPtr>(element);
      return;
    }
  }
  O2_BUILTIN_UNREACHABLE();
}

} // namespace o2::framework
