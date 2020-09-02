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
#include <string>
namespace o2::framework
{
/// This helper allows you to create a configurable option associated to a task.
/// Internally it will be bound to a ConfigParamSpec.
template <typename T>
struct Configurable {
  Configurable(std::string const& name, T defaultValue, std::string const& help)
    : name(name), value(defaultValue), help(help)
  {
  }
  using type = T;
  std::string name;
  T value;
  std::string help;
  operator T()
  {
    return value;
  }
  T const* operator->() const
  {
    return &value;
  }
};
template <typename T>
std::ostream& operator<<(std::ostream& os, Configurable<T> const& c)
{
  os << c.value;
  return os;
}
} // namespace o2::framework
#endif // O2_FRAMEWORK_CONFIGURABLE_H_
