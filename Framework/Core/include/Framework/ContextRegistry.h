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
  ContextRegistry();

  template <typename... Types>
  ContextRegistry(Types*... instances)
  {
    set(std::forward<Types*>(instances)...);
  }

  template <typename T>
  T* get() const
  {
    void* instance = nullptr;
    for (size_t i = 0; i < mRegistryCount; ++i) {
      if (mRegistryKey[i] == typeid(T*).hash_code()) {
        return reinterpret_cast<T*>(mRegistryValue[i]);
      }
    }
    throw std::out_of_range(std::string("Unsupported backend, no registered context '") + typeid(T).name() + "'");
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
    size_t i = 0;
    for (i = 0; i < mRegistryCount; ++i) {
      if (typeid(T*).hash_code() == mRegistryKey[i]) {
        return;
      }
    }
    if (i == MAX_REGISTRY_SIZE) {
      throw std::runtime_error("Too many entries in ContextRegistry");
    }
    mRegistryCount = i + 1;
    mRegistryKey[i] = typeid(T*).hash_code();
    mRegistryValue[i] = instance;
  }

 private:
  static constexpr size_t MAX_REGISTRY_SIZE = 8;
  size_t mRegistryCount = 0;
  std::array<size_t, MAX_REGISTRY_SIZE> mRegistryKey;
  std::array<void*, MAX_REGISTRY_SIZE> mRegistryValue;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CONTEXTREGISTRY_H
