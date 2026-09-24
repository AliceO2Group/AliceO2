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

/// \file GPUCommonTypeTraits.h
/// \author David Rohr

#ifndef GPUCOMMONTYPETRAITS_H
#define GPUCOMMONTYPETRAITS_H

#include "GPUCommonDef.h"

#if !defined(GPUCA_GPUCODE_DEVICE) || defined(__CUDACC__) || defined(__HIPCC__) // TODO: Get rid of GPUCommonTypeTraits once OpenCL supports <type_traits>
#ifndef GPUCA_GPUCODE_COMPILEKERNELS
#include <type_traits>
#endif
#else // OpenCL C++ and Metal, neither of which provides <type_traits>
// Spelled for MSL, which OpenCL C++ also accepts, so both backends share one implementation:
// enum values because program scope variables must be constant (MSL 4.1 spec, sec. 4.2), and
// is_pointer / is_member_pointer forward rather than inherit because MSL has no derived classes
// (sec. 1.5.4). Bare T* / T& need no address space: a partial specialization matches them all.
#ifdef __METAL__
#define GPUCA_TT_PROGRAMSCOPE constant
#else
#define GPUCA_TT_PROGRAMSCOPE // not empty-by-default: in OpenCL 'constant' would mean __constant
#endif
namespace std
{
template <bool B, class T, class F>
struct conditional {
  typedef T type;
};
template <class T, class F>
struct conditional<false, T, F> {
  typedef F type;
};
template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;

template <class T, class U>
struct is_same {
  enum { value = false };
};
template <class T>
struct is_same<T, T> {
  enum { value = true };
};
template <class T, class U>
GPUCA_TT_PROGRAMSCOPE static constexpr bool is_same_v = is_same<T, U>::value;

template <bool B, class T = void>
struct enable_if {
};
template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <class T>
struct remove_cv {
  typedef T type;
};
template <class T>
struct remove_cv<const T> {
  typedef T type;
};
template <class T>
struct remove_cv<volatile T> {
  typedef T type;
};
template <class T>
struct remove_cv<const volatile T> {
  typedef T type;
};
template <class T>
using remove_cv_t = typename remove_cv<T>::type;

template <class T>
struct remove_const {
  typedef T type;
};
template <class T>
struct remove_const<const T> {
  typedef T type;
};
template <class T>
using remove_const_t = typename remove_const<T>::type;

template <class T>
struct remove_volatile {
  typedef T type;
};
template <class T>
struct remove_volatile<volatile T> {
  typedef T type;
};
template <class T>
using remove_volatile_t = typename remove_volatile<T>::type;

template <class T>
struct is_pointer_t {
  enum { value = false };
};
template <class T>
struct is_pointer_t<T*> {
  enum { value = true };
};
template <class T>
struct is_pointer {
  enum { value = is_pointer_t<typename std::remove_cv<T>::type>::value };
};

template <class T>
struct remove_reference {
  typedef T type;
};
template <class T>
struct remove_reference<T&> {
  typedef T type;
};
template <class T>
struct remove_reference<T&&> {
  typedef T type;
};
template <class T>
using remove_reference_t = typename remove_reference<T>::type;

template <class T>
struct is_member_pointer_helper {
  enum { value = false };
};
template <class T, class U>
struct is_member_pointer_helper<T U::*> {
  enum { value = true };
};
template <class T>
struct is_member_pointer {
  enum { value = is_member_pointer_helper<typename std::remove_cv<T>::type>::value };
};
template <class T>
GPUCA_TT_PROGRAMSCOPE static constexpr bool is_member_pointer_v = is_member_pointer<T>::value;

} // namespace std
#undef GPUCA_TT_PROGRAMSCOPE
#endif

#endif
