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

/// @brief  pair of input and output size

#ifndef ALICEO2_CTFIO_SIZE_H
#define ALICEO2_CTFIO_SIZE_H

#include "GPUCommonRtypes.h"
#include <string>

namespace o2::ctf
{
struct CTFIOSize {
  size_t rawIn = 0;  // input size to compression
  size_t ctfIn = 0;  // input size to entropy encoding
  size_t ctfOut = 0; // encoded output

  bool operator==(const CTFIOSize& other) const { return (rawIn == other.rawIn) && (ctfIn == other.ctfIn) && (ctfOut == other.ctfOut); }
  bool operator!=(const CTFIOSize& other) const { return (rawIn != other.rawIn) || (ctfIn != other.ctfIn) || (ctfOut != other.ctfOut); }
  CTFIOSize operator+(const CTFIOSize& other) const { return {rawIn + other.rawIn, ctfIn + other.ctfIn, ctfOut + other.ctfOut}; }
  CTFIOSize operator-(const CTFIOSize& other) const { return {rawIn - other.rawIn, ctfIn + other.ctfIn, ctfOut - other.ctfOut}; }

  CTFIOSize& operator+=(const CTFIOSize& other)
  {
    rawIn += other.rawIn;
    ctfIn += other.ctfIn;
    ctfOut += other.ctfOut;
    return *this;
  }

  CTFIOSize& operator-=(const CTFIOSize& other)
  {
    rawIn -= other.rawIn;
    ctfIn -= other.ctfIn;
    ctfOut -= other.ctfOut;
    return *this;
  }

  std::string asString() const;

  ClassDefNV(CTFIOSize, 1);
};

} // namespace o2::ctf

#endif
