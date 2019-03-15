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

#include <unordered_map>
#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <string>
#include <stdexcept>

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
    try {
      instance = mRegistry.at(typeid(T).hash_code());
    } catch (std::out_of_range&) {
      throw std::out_of_range(std::string("Unsupported backend, no registered context '") + typeid(T).name() + "'");
    }
    return reinterpret_cast<T*>(instance);
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
    mRegistry[typeid(T).hash_code()] = instance;
  }

 private:
  std::unordered_map<size_t, void*> mRegistry;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CONTEXTREGISTRY_H
