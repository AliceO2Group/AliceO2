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
#include <memory>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <gsl/gsl>

namespace o2
{
namespace framework
{
/// Helper trait to determine if a given type T
/// is a specialization of a given reference type Ref.
/// See Framework/Core/test_TypeTraits.cxx for an example

template <typename T, template <typename...> class Ref>
struct is_specialization : std::false_type {
};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {
};

// helper struct to mark a type as non-messageable by defining a type alias
// with name 'non-messageable'
struct MarkAsNonMessageable {
};

// detect if a type is forced to be non-messageable, this is done by defining
// a type alias with name 'non-messageable' of the type MarkAsNonMessageable
template <typename T, typename _ = void>
struct is_forced_non_messageable : public std::false_type {
};

// specialization to detect the type a lias
template <typename T>
struct is_forced_non_messageable<
  T,
  typename std::enable_if<std::is_same<typename T::non_messageable, MarkAsNonMessageable>::value>::type> : public std::true_type {
};

// TODO: extend this to exclude structs with pointer data members
// see e.g. https://stackoverflow.com/questions/32880990/how-to-check-if-class-has-pointers-in-c14
template <typename T>
struct is_messageable : std::conditional<std::is_trivially_copyable<T>::value && //
                                           !std::is_polymorphic<T>::value &&     //
                                           !std::is_pointer<T>::value &&         //
                                           !is_forced_non_messageable<T>::value, //
                                         std::true_type,
                                         std::false_type>::type {
};

// FIXME: it apears that gsl:span matches the criteria for being messageable, regardless of the
// underlying type, but our goal is to identify structures that can be sent without serialization.
// needs investigation
template <typename T>
struct is_messageable<gsl::span<T>> : std::false_type {
};

// Detect a container by checking on the container properties
// this is the default trait implementation inheriting from false_type
template <typename T, typename _ = void>
struct is_container : std::false_type {
};

// helper to be substituted if the specified template arguments are
// available
template <typename... Ts>
struct class_member_checker {
};

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
      decltype(std::declval<T>().cend())>,
    void>> : public std::true_type {
};

// Detect whether a container class has a type definition `value_type` of messageable type
template <typename T, typename _ = void>
struct has_messageable_value_type : std::false_type {
};
template <typename T>
struct has_messageable_value_type<T, std::conditional_t<false, typename T::value_type, void>> : is_messageable<typename T::value_type> {
};

// Detect whether a class has a ROOT dictionary
// This member detector idiom is implemented using SFINAE idiom to look for
// a 'Class()' method.
// This is actually only covering classes using the ClasDef/Imp macros. ROOT
// serialization however is also possible for types only having the link
// in the LinkDef file. Such types can only be detected at runtime.
template <typename T, typename _ = void>
struct has_root_dictionary : std::false_type {
};

template <typename T>
struct has_root_dictionary<
  T,
  std::conditional_t<
    false,
    class_member_checker<
      decltype(std::declval<T>().Class())>,
    void>> : public std::true_type {
};

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

/// Helper class to deal with the case we are creating the first instance of a
/// (possibly) shared resource.
///
/// works for both:
///
/// std::shared_ptr<Base> storage = make_matching<decltype(storage), Concrete1>(args...);
///
/// or
///
/// std::unique_ptr<Base> storage = make_matching<decltype(storage), Concrete1>(args...);
///
/// Useful to deal with those who cannot make up their mind about ownership.
/// ;-)
template <typename HOLDER, typename T, typename... ARGS>
static std::enable_if_t<sizeof(std::declval<HOLDER>().unique()) != 0, HOLDER>
  make_matching(ARGS&&... args)
{
  return std::make_shared<T>(std::forward<ARGS>(args)...);
}

template <typename HOLDER, typename T, typename... ARGS>
static std::enable_if_t<sizeof(std::declval<HOLDER>().get_deleter()) != 0, HOLDER>
  make_matching(ARGS&&... args)
{
  return std::make_unique<T>(std::forward<ARGS>(args)...);
}

//Base case called by all overloads when needed. Derives from false_type.
template <typename Type, typename Archive = boost::archive::binary_oarchive, typename = std::void_t<>>
struct is_boost_serializable_base : std::false_type {
};

//Check if provided type implements a boost serialize method directly
template <class Type, typename Archive>
struct is_boost_serializable_base<Type, Archive,
                                  std::void_t<decltype(std::declval<Type&>().serialize(std::declval<Archive&>(), 0))>>
  : std::true_type {
};

// //Check if provided type is trivial (aka doesn't need a boost deserializer)
// template <class Type, typename Archive>
// struct is_boost_serializable_base<Type, Archive,
//                                   typename std::enable_if<boost::serialization::is_bitwise_serializable<typename Type::value_type>::value>::type>
//   : std::true_type {
// };

//Base implementation to provided recurrence. Wraps around base templates
template <class Type, typename Archive = boost::archive::binary_oarchive, typename = std::void_t<>>
struct is_boost_serializable
  : is_boost_serializable_base<Type, Archive> {
};

//Call base implementation in contained class/type if possible
template <class Type, typename Archive>
struct is_boost_serializable<Type, Archive, std::void_t<typename Type::value_type>>
  : is_boost_serializable<typename Type::value_type, Archive> {
};

//Call base implementation in contained class/type if possible. Added default archive type for convenience
template <class Type>
struct is_boost_serializable<Type, boost::archive::binary_oarchive, std::void_t<typename Type::value_type>>
  : is_boost_serializable<typename Type::value_type, boost::archive::binary_oarchive> {
};
} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TYPETRAITS_H
