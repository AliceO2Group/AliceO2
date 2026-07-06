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

#include <Framework/Traits.h>
#include <concepts>

namespace o2::aod
{
template <uint32_t>
struct Hash;
/// hash
/// 1. require aod::Hash
template <typename T>
concept is_aod_hash = requires(T t) { t.isHash(); };
/// 2. requires aod::Hash with header::DataOrigin
template <typename T>
concept is_origin_hash = requires(T t) { t.isOriginHash(); };

template <typename H>
struct MetadataTrait;
} // namespace o2::aod

namespace o2::soa
{
/// general
/// require a type to be not void
template <typename T>
concept not_void = requires { requires !std::same_as<T, void>; };

/// columns
/// 1. require a storage-backed column
template <typename C>
concept is_persistent_column = requires(C c) { c.isIteratableColumn(); };

/// 2. require self-index column
template <typename C>
concept is_self_index_column = not_void<typename std::decay_t<C>::self_index_t> && std::same_as<typename std::decay_t<C>::self_index_t, std::true_type>;

/// 3. require bidable index column
/// FIXME: this should really rely on the struct's content instead
struct Binding;
template <typename C>
concept is_index_column = !is_self_index_column<C> && requires(C c, o2::soa::Binding b) {
  { c.setCurrentRaw(b) } -> std::same_as<bool>;
  requires std::same_as<decltype(c.mBinding), o2::soa::Binding>;
};

/// 4. require a column that can be created from an expression
template <typename T>
concept is_spawnable_column = std::same_as<typename std::decay_t<T>::spawnable_t, std::true_type>;

/// 5. require an enumerating column, like soa::Index
template <typename C>
concept is_indexing_column = requires(C c) { c.isEnumeratingColumn(); };

/// 6. require a dynamic column
template <typename C>
concept is_dynamic_column = requires(C c) { c.isDynamicColumn(); };

/// 7. require a marking column
template <typename C>
concept is_marker_column = requires(C c) { c.isMarkingColumn(); };

/// 8. require any supported column
template <typename T>
concept is_column = is_persistent_column<T> || is_dynamic_column<T> || is_indexing_column<T> || is_marker_column<T>;

/// 9. require a type that can be bound as a column of type B
template <typename T, typename B>
concept can_bind = requires(T&& t) {
  { t.B::mColumnIterator };
};

/// 10. require at least one column in an exploded pack to be an indexing column
template <typename... C>
concept has_index = (is_indexing_column<C> || ...);

/// pack filtering helpers
template <typename T>
using is_dynamic_t = std::conditional_t<is_dynamic_column<T>, std::true_type, std::false_type>;

template <typename T>
using is_indexing_t = std::conditional_t<is_indexing_column<T>, std::true_type, std::false_type>;

/// tables, iterators and metadata
/// 1. require a type with parent_t dependent type
template <typename T>
concept has_parent_t = not_void<typename T::parent_t>;

/// 2. require a MetadataTrait specialization/descendant
/// FIXME: this should really rely on the struct's content instead
template <typename T>
concept is_metadata_trait = requires(T t) { t.isMetadataTrait(); };

/// 3. require a TableMetadata depcialization/descendant
/// FIXME: this should really rely on the struct's content instead
template <typename T>
concept is_metadata = requires(T t) { t.isTableMetadata(); };

/// 4. require a type with non-void metadata dependent type
template <typename T>
concept has_metadata = is_metadata<typename std::decay_t<T>::metadata>;

/// 5. require a type with non-void extension_table_t dependent type
template <typename T>
concept has_extension = is_metadata<T> && not_void<typename std::decay_t<T>::extension_table_t>;

/// 6. require a type with non-void configurable_t dependent type, that is same as true_type
template <typename T>
concept has_configurable_extension = has_extension<T> && requires(T t) { typename std::decay_t<T>::configurable_t; requires std::same_as<std::true_type, typename std::decay_t<T>::configurable_t>; };

/// 7. require an soa::Table
template <typename T>
concept is_table = requires(T t) { t.isSOATable(); };

/// 8. require a specialization/descendant of a TableIterator
template <typename T>
concept is_iterator = requires(T t) { t.isTableIterator(); };

/// 9. require a table or iterator
template <typename T>
concept is_table_or_iterator = is_table<T> || is_iterator<T>;

/// 10. require soa::IndexTable
/// FIXME: this should really rely on the struct's content instead
template <typename L, typename D, typename O, typename Key, typename H, typename... Ts>
struct IndexTable;

template <typename T>
concept is_index_table = framework::specialization_of_template<o2::soa::IndexTable, T>;

/// 11. require a type with a filtered policy
template <typename T>
concept has_filtered_policy = not_void<typename std::decay_t<T>::policy_t> && requires { std::decay_t<T>::policy_t::isFilteredIndexPolicy(); };

/// 12. require a filtered table iterator
template <typename T>
concept is_filtered_iterator = is_iterator<T> && has_filtered_policy<T>;

/// 13. require a filtered table
template <typename T>
concept is_filtered_table = requires(T t) { t.isFilteredBase(); };

/// 14. require a filtered table or iterator
template <typename T>
concept is_filtered = is_filtered_table<T> || is_filtered_iterator<T>;

/// 15. require not filtered table
template <typename T>
concept is_not_filtered_table = is_table<T> && !is_filtered_table<T>;

/// 16. require a join
template <typename T>
concept is_join = requires(T t) { t.isJoin(); };

/// 17. require an enumerated iterator
template <typename T>
concept is_enumerated_iterator = requires(T t) { t.globalIndex(); };

/// misc
/// 1. require a type with originals container
template <typename T>
concept with_originals = requires {
  T::originals.size();
};

/// 2. require a type with sources container
template <typename T>
concept with_sources = requires {
  T::sources.size();
};

/// 3. require a type with sources generator method
template <typename T>
concept with_sources_generator = requires(T t) {
  t.generateSources();
};

/// 4. require a type with ccd_urls container
template <typename T>
concept with_ccdb_urls = requires(T t) {
  t.ccdb_urls.size();
};

/// 5. require a type, whos metadata has base_table_t dependant type
/// FIXME: this should really rely on the struct's content instead
template <typename T>
concept with_base_table = with_originals<T> && has_metadata<aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>> && requires {
  typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata::base_table_t;
};

template <typename T>
concept with_base_table_ng = not_void<typename T::base_table_t>; // redicrection should be done at the check site

/// 6. require a type with expression_pack_t dependant type
template <typename T>
concept with_expression_pack = requires {
  typename T::expression_pack_t{};
};

/// 7. require a type with index_pack_t dependant type
template <typename T>
concept with_index_pack = requires {
  typename T::index_pack_t{};
};

/// 8. require SmallGroups
template <typename T>
concept is_smallgroups = requires(T t) { t.isSmallGroups(); };
} // namespace o2::soa

namespace o2::framework
{
/// preslice
/// 1. require a preslice policy
template <typename T>
concept is_preslice_policy = requires(T t) { t.isPreslicePolicy(); };

/// 2. require a preslice container
template <typename T>
concept is_preslice = requires(T t) { t.isPresliceContainer(); };

/// 3. reqiure a preslice group
template <typename T>
concept is_preslice_group = requires(T t) { t.isPresliceGroup(); };

/// 4. require a producable entity
template <typename T>
concept is_producable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::ref.desc_hash>>> || soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::parent_t::ref.desc_hash>>>;

/// 5. require produces declaration
template <typename T>
concept is_produces = requires(T t) { typename T::cursor_t; typename T::persistent_table_t; &T::cursor; };

/// 6. require produces group
template <typename T>
concept is_produces_group = requires(T t) { t.isProducesGroup(); };

/// 7. require spawnable entity
template <typename T>
concept is_spawnable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>> && soa::has_extension<typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata>;

/// 8. require dynamically spawnable entity
template <typename T>
concept is_dynamically_spawnable = soa::has_metadata<aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>> && soa::has_configurable_extension<typename aod::MetadataTrait<o2::aod::Hash<T::originals[T::originals.size() - 1].desc_hash>>::metadata>;

/// 9. require spawns declaration
template <typename T>
concept is_spawns = requires(T t) {
  typename T::metadata;
  typename T::expression_pack_t;
  t.projector.get();
};

/// 10. require defines declaration
template <typename T>
concept is_defines = requires(T t) {
  typename T::metadata;
  typename T::placeholders_pack_t;
  t.projector.get();
  requires std::same_as<decltype(t.needRecompilation), bool>;
  t.recompile();
};

/// 11. require builds declaration
template <typename T>
concept is_builds = requires(T t) {
  typename T::metadata;
  typename T::Key;
  t.map.size();
};

/// 12. require outputobj declaration
template <typename T>
concept is_outputobj = requires(T t) {
  &T::setHash;
  &T::spec;
  &T::ref;
  requires std::same_as<decltype(t.operator->()), typename T::obj_t*>;
  requires std::same_as<decltype(t.object.get()), typename T::obj_t*>;
};

/// 13. require service declaration
template <typename T>
concept is_service = requires(T t) {
  requires std::same_as<decltype(t.service), typename T::service_t*>;
  &T::operator->;
};

/// 14. require partition declaration
template <typename T>
concept is_partition = requires(T t) {
  &T::updatePlaceholders;
  t.mFiltered.get();
  &T::operator->;
  requires std::same_as<decltype(t.dataframeChanged), bool>;
  t.begin();
  t.end();
  t.size();
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_CONCEPTS_H
