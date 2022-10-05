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
#ifndef O2_FRAMEWORK_PACK_H_
#define O2_FRAMEWORK_PACK_H_

#include <cstddef>
#include <utility>

namespace o2::framework
{

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

#ifdef __clang__
template <std::size_t I, typename... Ts>
struct pack_element<I, pack<Ts...>> {
  using type = __type_pack_element<I, Ts...>;
};
#else

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
#endif

template <std::size_t I, typename T>
using pack_element_t = typename pack_element<I, T>::type;

template <typename T>
using pack_head_t = typename pack_element<0, T>::type;

template <typename Head, typename... Tail>
constexpr auto pack_tail(pack<Head, Tail...>)
{
  return pack<Tail...>{};
}

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
}

template <typename... Ps>
using concatenated_pack_t = decltype(concatenate_pack(Ps{}...));

template <typename... Args1, typename... Args2>
constexpr auto interleave_pack(pack<Args1...>, pack<Args2...>)
{
  return concatenated_pack_t<pack<Args1, Args2>...>{};
}

template <typename P1, typename P2>
using interleaved_pack_t = decltype(interleave_pack(P1{}, P2{}));

/// Selects from the pack types that satisfy the Condition
/// Multicondition takes the type to check as first template parameter
/// and any helper types as the following parameters
template <template <typename...> typename Condition, typename Result, typename... Cs>
constexpr auto select_pack(Result result, pack<>, pack<Cs...>)
{
  return result;
}

template <template <typename...> typename Condition, typename Result, typename T, typename... Cs, typename... Ts>
constexpr auto select_pack(Result result, pack<T, Ts...>, pack<Cs...> condPack)
{
  if constexpr (Condition<T, Cs...>()) {
    return select_pack<Condition>(concatenate_pack(result, pack<T>{}), pack<Ts...>{}, condPack);
  } else {
    return select_pack<Condition>(result, pack<Ts...>{}, condPack);
  }
}

template <template <typename...> typename Condition, typename... Types>
using selected_pack = std::decay_t<decltype(select_pack<Condition>(pack<>{}, pack<Types...>{}, pack<>{}))>;
template <template <typename...> typename Condition, typename CondPack, typename Pack>
using selected_pack_multicondition = std::decay_t<decltype(select_pack<Condition>(pack<>{}, Pack{}, CondPack{}))>;

/// Select only the items of a pack which match Condition
template <template <typename> typename Condition, typename Result>
constexpr auto filter_pack(Result result, pack<>)
{
  return result;
}

template <template <typename> typename Condition, typename Result, typename T, typename... Ts>
constexpr auto filter_pack(Result result, pack<T, Ts...>)
{
  if constexpr (Condition<T>()) {
    return filter_pack<Condition>(result, pack<Ts...>{});
  } else {
    return filter_pack<Condition>(concatenate_pack(result, pack<T>{}), pack<Ts...>{});
  }
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

template <template <typename, typename> typename Condition, typename T, typename Pack>
struct has_type_conditional;

template <template <typename, typename> typename Condition, typename T, typename... Us>
struct has_type_conditional<Condition, T, pack<Us...>> : std::disjunction<Condition<T, Us>...> {
};

template <template <typename, typename> typename Condition, typename T, typename... Us>
inline constexpr bool has_type_conditional_v = has_type_conditional<Condition, T, Us...>::value;

template <typename T>
constexpr size_t has_type_at(pack<> const&)
{
  return static_cast<size_t>(-1);
}

template <typename T, typename T1, typename... Ts>
constexpr size_t has_type_at(pack<T1, Ts...> const&)
{
  if constexpr (std::is_same_v<T, T1>) {
    return 0;
  } else if constexpr (has_type_v<T, pack<Ts...>>) {
    return 1 + has_type_at<T>(pack<Ts...>{});
  }
  return sizeof...(Ts) + 2;
}

template <template <typename, typename> typename Condition, typename T>
constexpr size_t has_type_at_conditional(pack<>&&)
{
  return static_cast<size_t>(-1);
}

template <template <typename, typename> typename Condition, typename T, typename T1, typename... Ts>
constexpr size_t has_type_at_conditional(pack<T1, Ts...>&&)
{
  if constexpr (Condition<T, T1>::value) {
    return 0;
  } else if constexpr (has_type_conditional_v<Condition, T, pack<Ts...>>) {
    return 1 + has_type_at_conditional<Condition, T>(pack<Ts...>{});
  }
  return sizeof...(Ts) + 2;
}

namespace
{
template <std::size_t I, typename T>
struct indexed {
  using type = T;
  constexpr static std::size_t index = I;
};

template <typename Is, typename... Ts>
struct indexer;

template <std::size_t... Is, typename... Ts>
struct indexer<std::index_sequence<Is...>, Ts...>
  : indexed<Is, Ts>... {
};

template <typename T, std::size_t I>
indexed<I, T> select(indexed<I, T>);

template <typename W, typename... Ts>
constexpr std::size_t has_type_at_t = decltype(select<W>(
  indexer<std::index_sequence_for<Ts...>, Ts...>{}))::index;
} // namespace

template <typename W>
constexpr std::size_t has_type_at_v(o2::framework::pack<>)
{
  return -1;
}

template <typename W, typename... Ts>
constexpr std::size_t has_type_at_v(o2::framework::pack<Ts...>)
{
  return has_type_at_t<W, Ts...>;
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

/// Subtract two packs
template <typename S1, typename S2>
struct subtract_pack {
  template <std::size_t... Indices>
  static constexpr auto make_subtraction(std::index_sequence<Indices...>)
  {
    return filtered_pack<std::is_void,
                         std::conditional_t<
                           !has_type_v<pack_element_t<Indices, S1>, S2>,
                           pack_element_t<Indices, S1>, void>...>{};
  }
  using type = decltype(make_subtraction(std::make_index_sequence<pack_size(S1{})>{}));
};

template <typename... Args1, typename... Args2>
constexpr auto concatenate_pack_unique(pack<Args1...>, pack<Args2...>)
{
  using p1 = typename subtract_pack<pack<Args1...>, pack<Args2...>>::type;
  return concatenate_pack(p1{}, pack<Args2...>{});
}

template <typename P1, typename P2, typename... Ps>
constexpr auto concatenate_pack_unique(P1 p1, P2 p2, Ps... ps)
{
  return concatenate_pack_unique(p1, concatenate_pack_unique(p2, ps...));
}

template <typename... Ps>
using concatenated_pack_unique_t = decltype(concatenate_pack_unique(Ps{}...));

template <typename PT>
constexpr auto unique_pack(pack<>, PT p2)
{
  return p2;
}

template <typename PT, typename T, typename... Ts>
constexpr auto unique_pack(pack<T, Ts...>, PT p2)
{
  return unique_pack(pack<Ts...>{}, concatenate_pack_unique(pack<T>{}, p2));
}

template <typename P>
using unique_pack_t = decltype(unique_pack(P{}, pack<>{}));

template <typename... Ts>
inline constexpr std::tuple<Ts...> pack_to_tuple(pack<Ts...>)
{
  return std::tuple<Ts...>{};
}

template <typename P>
using pack_to_tuple_t = decltype(pack_to_tuple(P{}));

template <typename T, std::size_t... Is>
inline auto sequence_to_pack(std::integer_sequence<std::size_t, Is...>)
{
  return pack<decltype((Is, T{}))...>{};
};

template <typename T, std::size_t N>
using repeated_type_pack_t = decltype(sequence_to_pack<T>(std::make_index_sequence<N>()));

} // namespace o2::framework

#endif // O2_FRAMEWORK_PACK_H_
