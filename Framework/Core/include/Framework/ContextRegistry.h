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
#include <stdexcept>
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
  inline ContextRegistry(Types*... instances);

  template <typename T>
  inline T* get() const;

  template <typename T, typename... Types>
  inline void set(T* instance, Types*... more);

  template <typename T>
  inline void set(T* element);

 private:
  static constexpr size_t MAX_REGISTRY_SIZE = 8;
  size_t mRegistryCount = 0;
  std::array<size_t, MAX_REGISTRY_SIZE> mRegistryKey;
  std::array<void*, MAX_REGISTRY_SIZE> mRegistryValue;
};

} // namespace o2::framework
#ifndef __CLING__
#include "Framework/ContextRegistry.inc"
#endif // __CLING__

#endif // O2_FRAMEWORK_CONTEXTREGISTRY_H_
