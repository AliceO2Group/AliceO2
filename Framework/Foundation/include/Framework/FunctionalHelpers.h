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
#ifndef O2_FRAMEWORK_FUNCTIONALHELPERS_H_
#define O2_FRAMEWORK_FUNCTIONALHELPERS_H_

#include "Framework/Pack.h"
#include <algorithm>
#include <functional>

namespace o2::framework
{

namespace
{
template <typename T>
struct memfun_type {
  using type = void;
};
} // namespace

template <size_t N>
struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N])
  {
    std::copy_n(str, N, value);
  }

  char value[N];
};

/// Type helper to hold metadata about a lambda or a class
/// method.
template <typename Ret, typename Class, typename... Args>
struct memfun_type<Ret (Class::*)(Args...) const> {
  using type = std::function<Ret(Args...)>;
  using args = pack<Args...>;
  using return_type = Ret;
};

/// Funtion From Lambda. Helper to create an std::function from a
/// lambda and therefore being able to use the std::function type
/// for template matching.
/// @return an std::function from a lambda (or anything actually callable). This
/// allows doing further template matching tricks to extract the arguments of the
/// function.
template <typename F>
typename memfun_type<decltype(&F::operator())>::type
  FFL(F const& func)
{
  return func;
}

/// @return metadata associated to method or a lambda.
template <typename F>
memfun_type<decltype(&F::operator())>
  FunctionMetadata(F const& func)
{
  return memfun_type<decltype(&F::operator())>();
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_FUNCTIONALHELPERS_H_
