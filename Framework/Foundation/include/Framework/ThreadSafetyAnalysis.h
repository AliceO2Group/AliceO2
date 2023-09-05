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
#ifndef O2_FRAMEWORK_THREADSAFETYANALYSIS_H_
#define O2_FRAMEWORK_THREADSAFETYANALYSIS_H_

/// Enable thread safety attributes only with clang.
/// The attributes can be safely erased when compiling with other compilers.
/// See https://clang.llvm.org/docs/ThreadSafetyAnalysis.html
/// for more information.
/// Moreover, since we do not want to wrap every single
/// std::mutex into our own class, we expect _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS
/// to be defined by the user to actually enable this feature.

// Assume that we only need this is someone else is not using the
// same clang provided macros
#ifndef O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__
#if defined(__clang__) && (!defined(SWIG)) && defined(_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS)
#define O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(x) // no-op
#endif

#define O2_DPL_CAPABILITY(x) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(capability(x))

#define O2_DPL_SCOPED_CAPABILITY \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

#define O2_DPL_GUARDED_BY(x) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))

#define O2_DPL_PT_GUARDED_BY(x) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))

#define O2_DPL_ACQUIRED_BEFORE(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define O2_DPL_ACQUIRED_AFTER(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define O2_DPL_REQUIRES(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(requires_capability(__VA_ARGS__))

#define O2_DPL_REQUIRES_SHARED(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))

#define O2_DPL_ACQUIRE(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define O2_DPL_ACQUIRE_SHARED(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define O2_DPL_RELEASE(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

#define O2_DPL_RELEASE_SHARED(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(release_shared_capability(__VA_ARGS__))

#define O2_DPL_RELEASE_GENERIC(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(release_generic_capability(__VA_ARGS__))

#define O2_DPL_TRY_ACQUIRE(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))

#define O2_DPL_TRY_ACQUIRE_SHARED(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))

#define O2_DPL_EXCLUDES(...) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define O2_DPL_ASSERT_CAPABILITY(x) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(assert_capability(x))

#define O2_DPL_ASSERT_SHARED_CAPABILITY(x) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_capability(x))

#define O2_DPL_RETURN_CAPABILITY(x) \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#define O2_DPL_NO_THREAD_SAFETY_ANALYSIS \
  O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

#endif // O2_DPL_THREAD_ANNOTATION_ATTRIBUTE__
#endif // O2_FRAMEWORK_THREADSAFETYANALYSIS_H_
