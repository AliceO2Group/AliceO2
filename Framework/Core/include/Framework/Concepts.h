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

#ifndef O2_FRAMEWORK_CONCEPTS_H
#define O2_FRAMEWORK_CONCEPTS_H

#include <concepts>

namespace o2::aod
{
/// hash
template <typename T>
concept is_aod_hash = requires(T t) { &T::isHash; };

template <typename T>
concept is_origin_hash = requires(T t) { &T::isOriginHash; };
} // namespace o2::aod

namespace o2::soa
{
/// general
template <typename T>
concept not_void = requires { requires !std::same_as<T, void>; };

/// columns
template <typename C>
concept is_persistent_column = requires(C c) { &C::isIteratableColumn; };

template <typename C>
concept is_self_index_column = not_void<typename C::self_index_t> && std::same_as<typename C::self_index_t, std::true_type>;

/// FIXME: this should really rely on the struct's content instead
struct Binding;
template <typename C>
concept is_index_column = !is_self_index_column<C> && requires(C c, o2::soa::Binding b) {
  { c.setCurrentRaw(b) } -> std::same_as<bool>;
  requires std::same_as<decltype(c.mBinding), o2::soa::Binding>;
};

template <typename T>
concept is_spawnable_column = std::same_as<typename T::spawnable_t, std::true_type>;

template <typename C>
concept is_indexing_column = requires(C c) { &C::isEnumeratingColumn; };

template <typename C>
concept is_dynamic_column = requires(C c) { &C::isDynamicColumn; };

template <typename C>
concept is_marker_column = requires { &C::isMarkingColumn; };

template <typename T>
concept is_column = is_persistent_column<T> || is_dynamic_column<T> || is_indexing_column<T> || is_marker_column<T>;

/// pack filtering helpers
template <typename T>
using is_dynamic_t = std::conditional_t<is_dynamic_column<T>, std::true_type, std::false_type>;

template <typename T>
using is_indexing_t = std::conditional_t<is_indexing_column<T>, std::true_type, std::false_type>;

/// tables, iterators and metadata
template <typename T>
concept has_parent_t = not_void<typename T::parent_t>;

/// FIXME: this should really rely on the struct's content instead
template <typename T>
concept is_metadata_trait = requires(T t) { &T::isMetadataTrait; };

/// FIXME: this should really rely on the struct's content instead
template <typename T>
concept is_metadata = requires(T t) { &T::isTableMetadata; };

template <typename T>
concept has_metadata = is_metadata_trait<T> && not_void<typename T::metadata>;

template <typename T>
concept has_extension = is_metadata<T> && not_void<typename T::extension_table_t>;

template <typename T>
concept has_configurable_extension = has_extension<T> && requires(T t) { typename T::configurable_t; requires std::same_as<std::true_type, typename T::configurable_t>; };

} // namespace o2::soa

namespace o2::framework
{

} // namespace o2::framework

#endif // O2_FRAMEWORK_CONCEPTS_H
