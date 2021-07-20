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

#ifndef O2_FRAMEWORK_STATICFOR_H_
#define O2_FRAMEWORK_STATICFOR_H_

namespace o2::framework
{
namespace staticFor_details
{
template <int FirstIndex, std::size_t... IndexSequence, typename F>
void applyFunction(F const& f, std::index_sequence<IndexSequence...>)
{
  (f(std::integral_constant<int, FirstIndex + IndexSequence>{}), ...);
}
} // namespace staticFor_details

template <int FirstIndex, int LastIndex, typename IndexSequence = std::make_index_sequence<(LastIndex - FirstIndex) + 1>, typename F>
static inline constexpr void static_for(F const& f)
{
  staticFor_details::applyFunction<FirstIndex>(f, IndexSequence{});
}
} // namespace o2::framework

#endif // O2_FRAMEWORK_STATICFOR_H_
