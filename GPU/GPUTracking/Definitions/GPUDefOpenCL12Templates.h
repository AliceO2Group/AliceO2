// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDefOpenCL12Templates.h
/// \author David Rohr, Sergey Gorbunov

// clang-format off
#ifndef GPUDEFOPENCL12TEMPLATES_H
#define GPUDEFOPENCL12TEMPLATES_H

#ifdef GPUCA_NOCOMPAT
  #define GPUCA_CPP11_INIT(...) __VA_ARGS__
#else
  #define GPUCA_CPP11_INIT(...)
#endif

//Special macros for OpenCL rev. 1.2 (encode address space in template parameter)
enum LocalOrGlobal { Mem_Local, Mem_Global, Mem_Constant, Mem_Plain };
#if defined(GPUCA_GPUCODE_DEVICE) && defined(GPUCA_USE_TEMPLATE_ADDRESS_SPACES)
  template<LocalOrGlobal, typename L, typename G, typename C, typename P> struct MakeTypeHelper;
  template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Local, L, G, C, P> { typedef L type; };
  template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Global, L, G, C, P> { typedef G type; };
  template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Constant, L, G, C, P> { typedef C type; };
  template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Plain, L, G, C, P> { typedef P type; };
  #define MakeType(base_type) typename MakeTypeHelper<LG, GPUsharedref() base_type, GPUglobalref() base_type, GPUconstantref() base_type, base_type>::type
  #define MEM_CLASS_PRE() template<LocalOrGlobal LG>
  #define MEM_CLASS_PRE_TEMPLATE(t) template<LocalOrGlobal LG, t>
  #define MEM_LG(type) type<LG>
  #define MEM_CLASS_PRE2() template<LocalOrGlobal LG2>
  #define MEM_CLASS_PRE2_TEMPLATE(t) template<LocalOrGlobal LG2, t>
  #define MEM_LG2(type) type<LG2>
  #define MEM_CLASS_PRE12() template<LocalOrGlobal LG> template<LocalOrGlobal LG2>
  #define MEM_CLASS_PRE23() template<LocalOrGlobal LG2, LocalOrGlobal LG3>
  #define MEM_LG3(type) type<LG3>
  #define MEM_CLASS_PRE234() template<LocalOrGlobal LG2, LocalOrGlobal LG3, LocalOrGlobal LG4>
  #define MEM_LG4(type) type<LG4>
  #define MEM_GLOBAL(type) type<Mem_Global>
  #define MEM_LOCAL(type) type<Mem_Local>
  #define MEM_LOCAL_TEMPLATE(type, t) type<Mem_Local, t>
  #define MEM_CONSTANT(type) type<Mem_Constant>
  #define MEM_PLAIN(type) type<Mem_Plain>
  #define MEM_TEMPLATE() template <typename T>
  #define MEM_TYPE(type) T
  #define MEM_TEMPLATE2() template <typename T, typename T2>
  #define MEM_TYPE2(type) T2
  #define MEM_TEMPLATE3() template <typename T, typename T2, typename T3>
  #define MEM_TYPE3(type) T3
  #define MEM_TEMPLATE4() template <typename T, typename T2, typename T3, typename T4>
  #define MEM_TYPE4(type) T4
#else
  #define MakeType(base_type) base_type
  #define MEM_CLASS_PRE()
  #define MEM_CLASS_PRE_TEMPLATE(t) template<t>
  #define MEM_LG(type) type
  #define MEM_CLASS_PRE2()
  #define MEM_CLASS_PRE2_TEMPLATE(t) template<t>
  #define MEM_LG2(type) type
  #define MEM_CLASS_PRE12()
  #define MEM_CLASS_PRE23()
  #define MEM_LG3(type) type
  #define MEM_CLASS_PRE234()
  #define MEM_LG4(type) type
  #define MEM_GLOBAL(type) type
  #define MEM_LOCAL(type) type
  #define MEM_LOCAL_TEMPLATE(type, t) type<t>
  #define MEM_CONSTANT(type) type
  #define MEM_PLAIN(type) type
  #define MEM_TEMPLATE()
  #define MEM_TYPE(type) type
  #define MEM_TEMPLATE2()
  #define MEM_TYPE2(type) type
  #define MEM_TEMPLATE3()
  #define MEM_TYPE3(type) type
  #define MEM_TEMPLATE4()
  #define MEM_TYPE4(type) type
#endif

#if defined(GPUCA_NO_CONSTANT_MEMORY) || defined(GPUCA_CONSTANT_AS_ARGUMENT)
  #undef MEM_CONSTANT
  #define MEM_CONSTANT(type) MEM_GLOBAL(type)
#endif

#endif //GPUDEFOPENCL12TEMPLATES_H
// clang-format on
