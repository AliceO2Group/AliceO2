// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_TYPETRAITS_H
#define FRAMEWORK_TYPETRAITS_H

#include <type_traits>
#include <vector>

namespace o2 {
namespace framework {
/// Helper trait to determine if a given type T
/// is a specialization of a given reference type Ref.
/// See Framework/Core/test_TypeTraits.cxx for an example

template<typename T, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

// helper struct to mark a type as non-messageable by defining a type alias
// with name 'non-messageable'
struct MarkAsNonMessageable {};

// detect if a type is forced to be non-messageable, this is done by defining
// a type alias with name 'non-messageable' of the type MarkAsNonMessageable
template< typename T, typename _ = void >
struct is_forced_non_messageable : public std::false_type {};

// specialization to detect the type a lias
template <typename T>
struct is_forced_non_messageable<
  T,
  typename std::enable_if<std::is_same<typename T::non_messageable, MarkAsNonMessageable>::value>::type
  > : public std::true_type {};

// TODO: extend this to exclude structs with pointer data members
// see e.g. https://stackoverflow.com/questions/32880990/how-to-check-if-class-has-pointers-in-c14
template <typename T>
struct is_messageable : std::conditional<std::is_trivially_copyable<T>::value &&
                                         !std::is_polymorphic<T>::value &&
                                         !is_forced_non_messageable<T>::value,
                                         std::true_type, std::false_type>::type {};

// Detect a container by checking on the container properties
// this is the default trait implementation inheriting from false_type
template <typename T, typename _ = void>
struct is_container : std::false_type {};

// helper to be substituted if the specified template arguments are
// available
template <typename... Ts>
struct class_member_checker {};

// the specialization for container types inheriting from true_type
// the helper can be substituted if all the specified members are available
// in the type
template <typename T>
struct is_container<
  T,
  std::conditional_t<
    false,
    class_member_checker<
      typename T::value_type,
      typename T::size_type,
      typename T::allocator_type,
      typename T::iterator,
      typename T::const_iterator,
      decltype(std::declval<T>().size()),
      decltype(std::declval<T>().begin()),
      decltype(std::declval<T>().end()),
      decltype(std::declval<T>().cbegin()),
      decltype(std::declval<T>().cend())
      >,
    void
    >
  > : public std::true_type {};


// Detect whether a class has a ROOT dictionary
// This member detector idiom is implemented using SFINAE idiom to look for
// a 'Class()' method.
// This is actually only covering classes using the ClasDef/Imp macros. ROOT
// serialization however is also possible for types only having the link
// in the LinkDef file. Such types can only be detected at runtime.
template <typename T, typename _ = void>
struct has_root_dictionary : std::false_type {};

template <typename T>
struct has_root_dictionary<
  T,
  std::conditional_t<
    false,
    class_member_checker<
      decltype(std::declval<T>().Class())
      >,
    void
    >
  > : public std::true_type {};

// specialization for containers
// covers cases with T::value_type having ROOT dictionary, meaning that
// std::map is not supported out of the box
//
// Note: this is only a SFINAE idiom if the default type of the second template
// argument in the declaration is the same as the conditionally produced one in
// the specialization (void in this case)
template <typename T>
class has_root_dictionary<T, typename std::enable_if<is_container<T>::value>::type>
  : public has_root_dictionary<typename T::value_type>
{
};

}
}
#endif // FRAMEWORK_TYPETRAITS_H
