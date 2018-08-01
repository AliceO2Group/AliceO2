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

#include <array>

namespace o2
{
namespace framework
{

/// Decouples getting the various contextes from the actual type
/// of context, so that the DataAllocator does not need to know
/// about the various serialization methods. Since there is only
/// a few context types we support, this can be done in an ad hoc
/// manner making sure each overload of ContextRegistry<T>::get()
/// uses a different entry in ContextRegistry::contextes;
///
/// Right now we use:
///
/// MessageContext 0
/// ROOTObjectContext 1
/// StringContext 2
class ContextRegistry
{
 public:
  ContextRegistry(std::array<void*, 2> contextes)
    : mContextes{ contextes }
  {
  }

  /// Default getter does nothing. Each Context needs
  /// to override the get method and return a unique
  /// entry in the mContextes.
  template <class T, size_t S = sizeof(T)>
  T* get()
  {
    static_assert(sizeof(T) == -1, "Unsupported backend");
  }

  /// Default setter does nothing. Each Context needs
  /// to override the set method and store the agreed
  /// pointer in the right position.
  template <class T, size_t S = sizeof(T)>
  void set(T*)
  {
    static_assert(sizeof(T) == -1, "Unsupported backend");
  }

 private:
  std::array<void*, 2> mContextes;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CONTEXTREGISTRY_H
