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

/// \file qGetLdBinarySymbols.h
/// \author David Rohr

#ifndef QGETLDBINARYSYMBOLS_H
#define QGETLDBINARYSYMBOLS_H

#define QGET_LD_BINARY_CAT3(a, b, c) a##b##c
#define QGET_LD_BINARY_SYMBOLS(filename)                             \
  extern "C" char QGET_LD_BINARY_CAT3(_binary_, filename, _start)[]; \
  extern "C" char QGET_LD_BINARY_CAT3(_binary_, filename, _end)[];   \
  static size_t QGET_LD_BINARY_CAT3(_binary_, filename, _len) = QGET_LD_BINARY_CAT3(_binary_, filename, _end) - QGET_LD_BINARY_CAT3(_binary_, filename, _start);

#endif // QGETLDBINARYSYMBOLS_H
