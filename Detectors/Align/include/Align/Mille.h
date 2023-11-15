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

/// @file   Mille.h
/// @author Gero Flucke, ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  October 2006
/// @brief  Class to write C binary file.

/**
 *  Class to write a C binary (cf. below) file of a given name and to fill it
 *  with information used as input to **pede**.
 *  Use its member functions \c mille(), \c special(), \c kill() and \c end()
 *  as you would use the fortran \ref mille.f90 "MILLE"
 *  and its entry points \c MILLSP, \c KILLE and \c ENDLE.
 *
 *  For debugging purposes constructor flags enable switching to text output and/or
 *  to write also derivatives and labels which are ==0.
 *  But note that **pede** will not be able to read text output and has not been tested with
 *  derivatives/labels ==0.
 */

#ifndef MILLE_H
#define MILLE_H

#include <fstream>
#include <TArrayI.h>
#include <TArrayF.h>

namespace o2
{
namespace align
{

class Mille
{
 public:
  Mille(const std::string& outFileName, bool asBinary = true, bool writeZero = false);
  void mille(int NLC, const float* derLc, int NGL, const float* derGl, const int* label, float rMeas, float sigma);
  void special(int nSpecial, const float* floatings, const int* integers);
  void clear();
  int finalise();
  void kill() { clear(); };        // alias to old Mille method
  int end() { return finalise(); } //  alias to old Mille method

 private:
  /// largest label allowed: 2^31 - 1
  static constexpr int MaxLabel = 0x7fffffff;
  std::ofstream mOutFile = {};     ///< C-binary for output
  bool mAsBinary = true;           ///< if false output as text
  bool mWriteZero = false;         ///< if true also write out derivatives/labels ==0
  bool mHasSpecial = false;        ///< if true, special(..) already called for this record
  std::vector<int> mBufferInt;     ///< to collect labels etc.
  std::vector<float> mBufferFloat; ///< to collect derivatives etc.
};

} // namespace align
} // namespace o2

#endif
