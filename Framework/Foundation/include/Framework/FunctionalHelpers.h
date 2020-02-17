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

namespace o2::framework
{

namespace
{
template <typename T>
struct memfun_type {
  using type = void;
};
} // namespace

/// Type helper to hold a parameter pack.  This is different from a tuple
/// as there is no data associated to it.
template <typename...>
struct pack {
};

/// template function to determine number of types in a pack
template <typename... Ts>
constexpr std::size_t pack_size(pack<Ts...> const&)
{
  return sizeof...(Ts);
}

template <std::size_t I, typename T>
struct pack_element;

// recursive case
template <std::size_t I, typename Head, typename... Tail>
struct pack_element<I, pack<Head, Tail...>>
  : pack_element<I - 1, pack<Tail...>> {
};

// base case
template <typename Head, typename... Tail>
struct pack_element<0, pack<Head, Tail...>> {
  typedef Head type;
};

template <std::size_t I, typename T>
using pack_element_t = typename pack_element<I, T>::type;

/// Templates for manipulating type lists in pack
/// (see https://codereview.stackexchange.com/questions/201209/filter-template-meta-function/201222#201222)
/// Example of use:
///     template<typename T>
///         struct is_not_double: std::true_type{};
///     template<>
///         struct is_not_double<double>: std::false_type{};
/// The following will return a pack, excluding double
///  filtered_pack<is_not_double, double, int, char, float*, double, char*, double>()
///
template <typename... Args1, typename... Args2>
constexpr auto concatenate_pack(pack<Args1...>, pack<Args2...>)
{
  return pack<Args1..., Args2...>{};
}

template <typename P1, typename P2, typename... Ps>
constexpr auto concatenate_pack(P1 p1, P2 p2, Ps... ps)
{
  return concatenate_pack(p1, concatenate_pack(p2, ps...));
};

template <typename... Ps>
using concatenated_pack_t = decltype(concatenate_pack(Ps{}...));

/// Selects from the pack types that satisfy the Condition
template <template <typename> typename Condition, typename Result>
constexpr auto select_pack(Result result, pack<>)
{
  return result;
}

template <template <typename> typename Condition, typename Result, typename T, typename... Ts>
constexpr auto select_pack(Result result, pack<T, Ts...>)
{
  if constexpr (Condition<T>())
    return select_pack<Condition>(concatenate_pack(result, pack<T>{}), pack<Ts...>{});
  else
    return select_pack<Condition>(result, pack<Ts...>{});
}

template <template <typename> typename Condition, typename... Types>
using selected_pack = std::decay_t<decltype(select_pack<Condition>(pack<>{}, pack<Types...>{}))>;

/// Select only the items of a pack which match Condition
template <template <typename> typename Condition, typename Result>
constexpr auto filter_pack(Result result, pack<>)
{
  return result;
}

template <template <typename> typename Condition, typename Result, typename T, typename... Ts>
constexpr auto filter_pack(Result result, pack<T, Ts...>)
{
  if constexpr (Condition<T>())
    return filter_pack<Condition>(result, pack<Ts...>{});
  else
    return filter_pack<Condition>(concatenate_pack(result, pack<T>{}), pack<Ts...>{});
}

template <typename T>
void print_pack()
{
  puts(__PRETTY_FUNCTION__);
}

template <template <typename> typename Condition, typename... Types>
using filtered_pack = std::decay_t<decltype(filter_pack<Condition>(pack<>{}, pack<Types...>{}))>;

/// Check if a given pack Pack has a type T inside.
template <typename T, typename Pack>
struct has_type;

template <typename T, typename... Us>
struct has_type<T, pack<Us...>> : std::disjunction<std::is_same<T, Us>...> {
};

template <typename T, typename... Us>
inline constexpr bool has_type_v = has_type<T, Us...>::value;

template <typename T>
constexpr size_t has_type_at(pack<> const&)
{
  return static_cast<size_t>(-1);
}

template <typename T, typename T1, typename... Ts>
constexpr size_t has_type_at(pack<T1, Ts...> const&)
{
  if constexpr (std::is_same_v<T, T1>)
    return 0;
  if constexpr (has_type_v<T, pack<T1, Ts...>>)
    return 1 + has_type_at<T>(pack<Ts...>{});
  return sizeof...(Ts) + 2;
}

/// Intersect two packs
template <typename S1, typename S2>
struct intersect_pack {
  template <std::size_t... Indices>
  static constexpr auto make_intersection(std::index_sequence<Indices...>)
  {
    return filtered_pack<std::is_void,
                         std::conditional_t<
                           has_type_v<pack_element_t<Indices, S1>, S2>,
                           pack_element_t<Indices, S1>, void>...>{};
  }
  using type = decltype(make_intersection(std::make_index_sequence<pack_size(S1{})>{}));
};

template <typename S1, typename S2>
using intersected_pack_t = typename intersect_pack<S1, S2>::type;

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

#endif // o2_framework_FunctionalHelpers_H_INCLUDED
