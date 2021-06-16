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

// Compatibility layer for a few common builtins found in different compilers.
// These are trivial and the goal is to make sure we can compile on non GCC /
// clang compilers if requested.
#ifndef O2_FRAMEWORK_COMPILER_BUILTINS_H_
#define O2_FRAMEWORK_COMPILER_BUILTINS_H_
#if __GNUC__
#define O2_BUILTIN_UNREACHABLE __builtin_unreachable
#elif __clang__
#define O2_BUILTIN_UNREACHABLE __builtin_unreachable
#else
#define O2_BUILTIN_UNREACHABLE
#endif

#if __GNUC__
#define O2_BUILTIN_LIKELY(x) __builtin_expect((x), 1)
#define O2_BUILTIN_UNLIKELY(x) __builtin_expect((x), 0)
#elif __clang__
#define O2_BUILTIN_LIKELY(x) __builtin_expect((x), 1)
#define O2_BUILTIN_UNLIKELY(x) __builtin_expect((x), 0)
#else
#define O2_BUILTIN_LIKELY(x)
#define O2_BUILTIN_UNLIKELY(x)
#endif

#if __GNUC__
#define O2_BUILTIN_PREFETCH(x, ...) __builtin_prefetch((x), __VA_ARGS__)
#elif __clang__
#define O2_BUILTIN_PREFETCH(x, ...) __builtin_prefetch((x), __VA_ARGS__)
#else
#define O2_BUILTIN_PREFETCH(x, ...)
#endif

#if __GNUC__
#define O2_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#elif __clang__
#define O2_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#else
#define O2_VISIBILITY_HIDDEN
#endif

#endif // O2_FRAMEWORK_COMPILER_BUILTINS_H_
