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
///
/// \file ArrayUtils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_ARRAYUTILS_H_
#define TRACKINGITSU_INCLUDE_ARRAYUTILS_H_

#include <array>
#include <cstddef>
#include <utility>

namespace o2
{
namespace its
{
namespace CA
{

namespace ArrayUtils
{
template <typename T, std::size_t... Is, typename Initializer>
constexpr std::array<T, sizeof...(Is)> fillArray(Initializer, std::index_sequence<Is...>);
template <typename T, std::size_t N, typename Initializer>
constexpr std::array<T, N> fillArray(Initializer);
} // namespace ArrayUtils

template <typename T, std::size_t... Is, typename Initializer>
constexpr std::array<T, sizeof...(Is)> ArrayUtils::fillArray(Initializer initializer, std::index_sequence<Is...>)
{
  return std::array<T, sizeof...(Is)>{{initializer(Is)...}};
}

template <typename T, std::size_t N, typename Initializer>
constexpr std::array<T, N> ArrayUtils::fillArray(Initializer initializer)
{
  return ArrayUtils::fillArray<T>(initializer, std::make_index_sequence<N>{});
}
} // namespace CA
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_ARRAYUTILS_H_ */
