// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CONFIGURABLE_H_
#define O2_FRAMEWORK_CONFIGURABLE_H_
#include "Framework/ConfigurableKinds.h"
#include <string>
#include <vector>
namespace o2::framework
{
template <typename T, ConfigParamKind K>
struct ConfigurableBase {
  ConfigurableBase(std::string const& name, T&& defaultValue, std::string const& help)
    : name(name), value{std::forward<T>(defaultValue)}, help(help)
  {
  }
  using type = T;
  std::string name;
  T value;
  std::string help;
  static constexpr ConfigParamKind kind = K;
};

template <typename T, ConfigParamKind K>
struct ConfigurablePolicyConst : ConfigurableBase<T, K> {
  ConfigurablePolicyConst(std::string const& name, T&& defaultValue, std::string const& help)
    : ConfigurableBase<T, K>{name, std::forward<T>(defaultValue), help}
  {
  }
  operator T()
  {
    return this->value;
  }
  T const* operator->() const
  {
    return &this->value;
  }
};

template <typename T, ConfigParamKind K>
struct ConfigurablePolicyMutable : ConfigurableBase<T, K> {
  ConfigurablePolicyMutable(std::string const& name, T&& defaultValue, std::string const& help)
    : ConfigurableBase<T, K>{name, std::forward<T>(defaultValue), help}
  {
  }
  operator T()
  {
    return this->value;
  }
  T* operator->()
  {
    return &this->value;
  }
};

/// This helper allows you to create a configurable option associated to a task.
/// Internally it will be bound to a ConfigParamSpec.
template <typename T, ConfigParamKind K = ConfigParamKind::kGeneric, typename IP = ConfigurablePolicyConst<T, K>>
struct Configurable : IP {
  Configurable(std::string const& name, T&& defaultValue, std::string const& help)
    : IP{name, std::forward<T>(defaultValue), help}
  {
  }
};

template <typename T, ConfigParamKind K = ConfigParamKind::kGeneric>
using MutableConfigurable = Configurable<T, K, ConfigurablePolicyMutable<T, K>>;

using ConfigurableAxis = Configurable<std::vector<double>, ConfigParamKind::kAxisSpec, ConfigurablePolicyConst<std::vector<double>, ConfigParamKind::kAxisSpec>>;

template <typename T, ConfigParamKind K, typename IP>
std::ostream& operator<<(std::ostream& os, Configurable<T, K, IP> const& c)
{
  os << c.value;
  return os;
}
} // namespace o2::framework
#endif // O2_FRAMEWORK_CONFIGURABLE_H_
