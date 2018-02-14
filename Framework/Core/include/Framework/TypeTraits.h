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

namespace o2 {
namespace framework {
/// Helper trait to determine if a given type T
/// is a specialization of a given reference type Ref.
/// See Framework/Core/test_TypeTraits.cxx for an example

template<typename T, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

// TODO: extend this to exclude structs with pointer data members
// see e.g. https://stackoverflow.com/questions/32880990/how-to-check-if-class-has-pointers-in-c14
template <typename T>
struct is_messageable : std::conditional<std::is_trivially_copyable<T>::value && !std::is_polymorphic<T>::value,
                                         std::true_type, std::false_type>::type {};
}
}
#endif // FRAMEWORK_TYPETRAITS_H
