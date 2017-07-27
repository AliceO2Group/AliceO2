// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_SERVICEREGISTRY_H
#define FRAMEWORK_SERVICEREGISTRY_H

#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

namespace o2 {
namespace framework {

// FIXME: decide what we really want from this
class ServiceRegistry
{
public:
  // Register a service for the given interface T
  // with actual implementation C, i.e. C is derived from T.
  // Only one instance of type C can be registered per type T.
  // The fact we use a bare pointer indicates that the ownership
  // of the service still belongs to whatever created it, and is
  // not passed to the registry. It's therefore responsibility of
  // the creator of the service to properly dispose it.
  template<class I, class C>
  void registerService(C *service) {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in 
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    mServices[typeid(I)] = reinterpret_cast<ServicePtr>(service);
  }

  // Get a service for the given interface T. The returned reference
  // exposed to the user.
  // is actually of the last concrete type C, however this should not be
  template<typename T>
  T &get() {
    auto service = mServices.find(typeid(T));
    if (service == mServices.end()) {
      throw std::runtime_error(std::string("Unable to find service of kind ") +
                               typeid(T).name() +
                               "did you register one?");
    }
    return *reinterpret_cast<T*>(service->second);
  }
private:
  using TypeInfoRef = std::reference_wrapper<const std::type_info>;
  using ServicePtr = void *;
  struct Hasher {
      std::size_t operator()(TypeInfoRef code) const
      {
          return code.get().hash_code();
      }
  };
  struct EqualTo {
      bool operator()(TypeInfoRef lhs, TypeInfoRef rhs) const
      {
          return lhs.get() == rhs.get();
      }
  };
  std::unordered_map<TypeInfoRef, ServicePtr, Hasher, EqualTo> mServices;
};

} // namespace framework
} // namespace o2

#endif //FRAMEWORK_SERVICEREGISTRY_H
