// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_IMPL_HELPERS_MAKE_ARRAY_H
#define O2_MCH_RAW_IMPL_HELPERS_MAKE_ARRAY_H

#include <array>
#include <type_traits>

namespace o2::mch::raw::impl
{

template <typename CTOR, size_t... S>
std::array<std::invoke_result_t<CTOR, size_t>, sizeof...(S)> makeArray(CTOR&& ctor,
                                                                       std::index_sequence<S...>)
{
  return std::array<std::invoke_result_t<CTOR, size_t>, sizeof...(S)>{{ctor(S)...}};
}

template <size_t N, typename CTOR>
std::array<std::invoke_result_t<CTOR, size_t>, N> makeArray(CTOR&& ctor)
{
  return makeArray(std::forward<CTOR>(ctor), std::make_index_sequence<N>());
}

} // namespace o2::mch::raw::impl
#endif
