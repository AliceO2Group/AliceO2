// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CHECKTYPES_H_
#define O2_FRAMEWORK_CHECKTYPES_H_

namespace o2::framework
{

/// Helper to understand if a given type is complete (declared fully) or not (forward declared).
/// See also: https://devblogs.microsoft.com/oldnewthing/20190710-00/?p=102678
template <typename, typename = void>
constexpr bool is_type_complete_v = false;

template <typename T>
constexpr bool is_type_complete_v<T, std::void_t<decltype(sizeof(T))>> = true;

/// Helper which will invoke lambda if the type T is actually available.
/// Can be used to check for existence or not of a given type.
template <typename T, typename TLambda>
void call_if_defined(TLambda&& lambda)
{
  if constexpr (is_type_complete_v<T>) {
    lambda(static_cast<T*>(nullptr));
  }
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHECKTYPES_H_
