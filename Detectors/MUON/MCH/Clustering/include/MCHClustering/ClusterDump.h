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

/// \file ClusterFindergem.h
/// \brief Definition of a class to reconstruct clusters with the gem MLEM algorithm
///
/// \author GG, Subatech

#ifndef ALICEO2_MCH_CLUSTERDUMP_H_
#define ALICEO2_MCH_CLUSTERDUMP_H_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// GG
#include <fstream>

namespace o2
{
namespace mch
{
// Utilities to dump a file
class ClusterDump
{
 public:
  ClusterDump(const char* str, int mode);
  ~ClusterDump();

  ClusterDump(const ClusterDump&) = delete;
  ClusterDump& operator=(const ClusterDump&) = delete;
  ClusterDump(ClusterDump&&) = delete;
  ClusterDump& operator=(ClusterDump&&) = delete;

  void flush();

  void dumpFloat32(int ifile, long size, const float_t* data);

  void dumpFloat64(int ifile, long size, const double_t* data);

  void dumpInt32(int ifile, long size, const int32_t* data);

  void dumpUInt32(int ifile, long size, const uint32_t* data);

  void dumpInt16(int ifile, long size, const int16_t* data);

  const std::string& getName() const { return fileName; };

 private:
  std::string fileName;
  std::fstream dumpFiles[1];
  int mode; // 1 write the file else don't
};
} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_CLUSTERDUMP_H_
