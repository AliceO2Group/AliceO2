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
#ifndef O2_FRAMEWORK_CONFIGURABLE_H_
#define O2_FRAMEWORK_CONFIGURABLE_H_
#include "Framework/ConfigurableKinds.h"
#include "Framework/Traits.h"
#include <string>
#include <vector>
namespace o2::framework
{
namespace expressions
{
struct PlaceholderNode;
}

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
  auto node()
  {
    return expressions::PlaceholderNode{*this};
  }
};

template <typename T, ConfigParamKind K = ConfigParamKind::kGeneric>
using MutableConfigurable = Configurable<T, K, ConfigurablePolicyMutable<T, K>>;

/// Convenience wrapper for overriding the CCDB path of a CCDB column declared
/// with DECLARE_SOA_CCDB_COLUMN / DECLARE_SOA_CCDB_COLUMN_FULL.
///
/// The option name, default value, and help string are all derived automatically
/// from the column type: name = "ccdb:" + Column::mLabel, default = Column::query.
///
/// Example:
///   struct MyTask {
///     ConfigurableCCDBPath<tofcalib::LHCphase> lhcPhasePath;
///   };
template <typename Column>
struct ConfigurableCCDBPath : Configurable<std::string> {
  ConfigurableCCDBPath()
    : Configurable<std::string>{std::string{"ccdb:"} + Column::mLabel,
                                std::string{Column::query},
                                std::string{"CCDB path for "} + Column::mLabel + " (default: " + Column::query + ")"}
  {
  }
};

template <typename T>
concept is_configurable = requires(T t) {
  requires std::same_as<std::string, decltype(t.name)>;
  requires std::same_as<std::string, decltype(t.help)>;
  requires std::same_as<typename std::decay_t<T>::type, decltype(t.value)>;
};

using ConfigurableAxis = Configurable<std::vector<double>, ConfigParamKind::kAxisSpec, ConfigurablePolicyConst<std::vector<double>, ConfigParamKind::kAxisSpec>>;

template <typename T>
concept is_configurable_axis = is_configurable<T> &&
                               requires() {
                                 T::kind == ConfigParamKind::kAxisSpec;
                               };

template <typename T, typename... As>
struct ProcessConfigurable : Configurable<bool, ConfigParamKind::kProcessFlag> {
  ProcessConfigurable(void (T::*process_)(As...), std::string const& name_, bool&& value_, std::string const& help_)
    : process{process_},
      Configurable<bool, ConfigParamKind::kProcessFlag>(name_, std::forward<bool>(value_), help_)
  {
  }
  void (T::*process)(As...);
};

template <typename T>
concept is_process_configurable = is_configurable<T> && requires(T t) { t.process; };

#define PROCESS_SWITCH(_Class_, _Name_, _Help_, _Default_) \
  decltype(o2::framework::ProcessConfigurable{&_Class_ ::_Name_, #_Name_, _Default_, _Help_}) do##_Name_ = o2::framework::ProcessConfigurable{&_Class_ ::_Name_, #_Name_, _Default_, _Help_};
#define PROCESS_SWITCH_FULL(_Class_, _Method_, _Name_, _Help_, _Default_) \
  decltype(o2::framework::ProcessConfigurable{&_Class_ ::_Method_, #_Name_, _Default_, _Help_}) do##_Name_ = o2::framework::ProcessConfigurable{&_Class_ ::_Method_, #_Name_, _Default_, _Help_};

template <typename T, ConfigParamKind K, typename IP>
std::ostream& operator<<(std::ostream& os, Configurable<T, K, IP> const& c)
{
  os << c.value;
  return os;
}

/// Can be used to group together a number of Configurables
/// to overcome the limit of 100 Configurables per task.
/// In order to do so you can do:
///
/// struct MyTask {
///   struct MyGroup : ConfigurableGroup {
///     Configurable<int> aCut{...};
///     Configurable<float> bCut{...};
///   } group;
/// };
///
/// and access it with
///
/// group.aCut;
struct ConfigurableGroup {
};

template <typename T>
concept is_configurable_group = std::derived_from<T, ConfigurableGroup>;

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONFIGURABLE_H_
