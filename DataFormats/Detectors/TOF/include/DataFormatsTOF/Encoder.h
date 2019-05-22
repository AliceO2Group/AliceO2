// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Encoder.h
/// \brief Definition of the TOF encoder

#ifndef ALICEO2_TOF_ENCODER_H
#define ALICEO2_TOF_ENCODER_H

#include <fstream>
#include <string>
#include <cstdint>
#include "DataFormat.h"

namespace o2
{
namespace tof
{
namespace compressed
{
/// \class Encoder
/// \brief Encoder class for TOF
///
class Encoder
{

 public:
  Encoder() = default;
  ~Encoder() = default;

  bool open(std::string name);
  bool alloc(long size);
  bool encode(/*define input structure*/);
  bool flush();
  bool close();
  void setVerbose(bool val) { mVerbose = val; };

  // benchmarks
  double mIntegratedBytes = 0.;
  double mIntegratedTime = 0.;

 protected:
  std::ofstream mFile;
  bool mVerbose = false;

  char* mBuffer = nullptr;
  long mSize;
  Union_t* mUnion = nullptr;
};

} // namespace compressed
} // namespace tof
} // namespace o2
#endif
