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
#ifndef FRAMEWORK_CONFIGPARAMSPEC_H
#define FRAMEWORK_CONFIGPARAMSPEC_H

#include "Framework/Variant.h"
#include "Framework/ConfigurableKinds.h"

#include <string>

namespace o2::framework
{
/// @class ConfigParamSpec Definition of options for a processor
/// An option definition consists of a name, a type, a help message, and
/// an optional default value. The type of the argument has to be specified
/// in terms of VariantType
///
/// Options and arguments can be retrieved from the init context in the
/// initialization function:
///   context.options().get<TYPE>("NAME")
///
/// All options are also forwarded to the device.
struct ConfigParamSpec {
  using ParamType = VariantType;

  struct HelpString {
    const char* c_str() const { return str.c_str(); }
    std::string str;
  };
  template <typename T>
  ConfigParamSpec(std::string, ParamType, Variant, T)
    : type(VariantType::Unknown), kind{ConfigParamKind::kGeneric}
  {
    static_assert(std::is_same<T, HelpString>::value,
                  R"(help string must be brace-enclosed, e.g. '{"help"}')");
  }

  ConfigParamSpec(std::string _name, ParamType _type,
                  Variant _defaultValue, HelpString _help, ConfigParamKind kind_ = ConfigParamKind::kGeneric)
    : name(_name), type(_type), defaultValue(_defaultValue), help(_help), kind{kind_}
  {
  }

  /// config spec without default value, explicitely marked as 'empty'
  /// and will be ignored at other places
  ConfigParamSpec(std::string _name, ParamType _type, HelpString _help)
    : name(_name), type(_type), defaultValue(VariantType::Empty), help(_help), kind{ConfigParamKind::kGeneric} {}

  bool operator==(ConfigParamSpec const& that) const
  {
    return name == that.name && type == that.type;
  }

  std::string name;
  ParamType type;
  Variant defaultValue;
  HelpString help;
  ConfigParamKind kind;
};

} // namespace o2::framework

#endif // FRAMEWORK_CONFIGPARAMSPEC_H
