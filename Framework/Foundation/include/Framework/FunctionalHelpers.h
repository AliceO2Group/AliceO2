// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_FunctionalHelpers_H_INCLUDED
#define o2_framework_FunctionalHelpers_H_INCLUDED

#include <functional>

namespace o2
{
namespace framework
{

namespace
{
template <typename T>
struct memfun_type {
  using type = void;
};

/// Type helper to hold a parameter pack.  This is different from a tuple
/// as there is no data associated to it.
template <typename...>
struct pack {
};

/// Templates for manipulating type lists in pack
/// (see https://codereview.stackexchange.com/questions/201209/filter-template-meta-function/201222#201222)
/// Example of use:
///     template<typename T>
///         struct is_not_double: std::true_type{};
///     template<>
///         struct is_not_double<double>: std::false_type{};
/// The following will return a pack, excluding double
///  <filtered_pack<is_not_double, double, int, char, float*, double, char*, double>>()
///
template <typename... Args1, typename... Args2>
constexpr auto concatenate_pack(pack<Args1...>, pack<Args2...>)
{
  return pack<Args1..., Args2...>{};
}

template <template <typename> typename Condition, typename Result>
constexpr auto filter_pack(Result result, pack<>)
{
  return result;
}

template <template <typename> typename Condition, typename Result, typename T, typename... Ts>
constexpr auto filter_pack(Result result, pack<T, Ts...>)
{
  if constexpr (Condition<T>{})
    return filter_pack<Condition>(concatenate_pack(result, pack<T>{}), pack<Ts...>{});
  else
    return filter_pack<Condition>(result, pack<Ts...>{});
}

template <template <typename> typename Condition, typename... Types>
using filtered_pack = std::decay_t<decltype(filter_pack<Condition>(pack<>{}, pack<Types...>{}))>;

/// Type helper to hold metadata about a lambda or a class
/// method.
template <typename Ret, typename Class, typename... Args>
struct memfun_type<Ret (Class::*)(Args...) const> {
  using type = std::function<Ret(Args...)>;
  using args = pack<Args...>;
  using return_type = Ret;
};
} // namespace

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

} // namespace framework
} // namespace o2

#endif // o2_framework_FunctionalHelpers_H_INCLUDED
