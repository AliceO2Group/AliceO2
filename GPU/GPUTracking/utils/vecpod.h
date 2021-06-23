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

/// \file vecpod.h
/// \author David Rohr

#include <vector>

template <class T>
struct vecpod_allocator {
  typedef T value_type;
  vecpod_allocator() noexcept : stdalloc() {}
  T* allocate(std::size_t n) { return stdalloc.allocate(n); }
  void deallocate(T* p, std::size_t n) { stdalloc.deallocate(p, n); }
  static void construct(T*) {}
  std::allocator<T> stdalloc;
};

template <class T>
using vecpod = typename std::vector<T, vecpod_allocator<T>>;
// template <class T> using vecpod = typename std::vector<T>;
