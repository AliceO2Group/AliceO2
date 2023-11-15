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

/*
  RS: original Mille.cc from http://svnsrv.desy.de/public/MillepedeII/tags/V04-02-03
*/

#include "Align/Mille.h"
#include "Framework/Logger.h"
#include <fstream>
#include <iostream>

namespace o2
{
namespace align
{
//___________________________________________________________________________
Mille::Mille(const std::string& outFileName, bool asBinary, bool writeZero)
  : mOutFile(outFileName, (asBinary ? (std::ios::binary | std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::trunc))),
    mAsBinary(asBinary),
    mWriteZero(writeZero)
{
  if (!mOutFile.is_open()) {
    LOG(fatal) << "Failed to open Mille file " << outFileName;
  }
  mBufferInt.reserve(1024);
  mBufferFloat.reserve(1024);
  clear();
}

//___________________________________________________________________________
/// Add measurement to buffer.
/**
 * \param[in]    NLC    number of local derivatives
 * \param[in]    derLc  local derivatives
 * \param[in]    NGL    number of global derivatives
 * \param[in]    derGl  global derivatives
 * \param[in]    label  global labels
 * \param[in]    rMeas  measurement (residuum)
 * \param[in]    sigma  error
 */
void Mille::mille(int NLC, const float* derLc,
                  int NGL, const float* derGl, const int* label,
                  float rMeas, float sigma)
{
  if (sigma <= 0.) {
    return;
  }
  // first store measurement
  mBufferFloat.push_back(rMeas);
  mBufferInt.push_back(0);

  // store local derivatives and local 'lables' 1,...,NLC
  for (int i = 0; i < NLC; ++i) {
    if (derLc[i] || mWriteZero) {       // by default store only non-zero derivatives
      mBufferFloat.push_back(derLc[i]); // local derivatives
      mBufferInt.push_back(i + 1);      // index of local parameter
    }
  }

  // store uncertainty of measurement in between locals and globals
  mBufferFloat.push_back(sigma);
  mBufferInt.push_back(0);

  // store global derivatives and their labels
  for (int i = 0; i < NGL; ++i) {
    if (derGl[i] || mWriteZero) {                                 // by default store only non-zero derivatives
      if ((label[i] > 0 || mWriteZero) && label[i] <= MaxLabel) { // and for valid labels
        mBufferFloat.push_back(derGl[i]);                         // global derivatives
        mBufferInt.push_back(label[i]);                           // index of global parameter
      } else {
        LOGP(error, "Mille: invalid label {} <=0 or > {}", label[i], MaxLabel);
      }
    }
  }
}

//___________________________________________________________________________
/// Add special data to buffer.
/**
 * \param[in]    nSpecial   number of floats/ints
 * \param[in]    floats  floats
 * \param[in]    ints   ints
 */
void Mille::special(int nSpecial, const float* floats, const int* ints)
{
  if (nSpecial == 0) {
    return;
  }
  if (mHasSpecial) {
    LOG(error) << "Mille: special values already stored for this record.";
    return;
  }
  mHasSpecial = true;
  //  mBufferFloat[.]  | mBufferInt[.]
  // ------------------------------------
  //      0.0           |      0
  //  -float(nSpecial)  |      0
  //  The above indicates special data, following are nSpecial floating and nSpecial integer data.
  //
  // zero pair
  mBufferFloat.push_back(0.);
  mBufferInt.push_back(0);
  // nSpecial and zero
  mBufferFloat.push_back(-nSpecial); // automatic conversion to float
  mBufferInt.push_back(0);

  for (int i = 0; i < nSpecial; ++i) {
    mBufferFloat.push_back(floats[i]);
    mBufferInt.push_back(ints[i]);
  }
}

//___________________________________________________________________________
/// Write buffer (set of derivatives with same local parameters) to file.
int Mille::finalise()
{
  int wrote = 0;
  int nw = mBufferInt.size();
  if (nw) { // only if anything stored...
    const int numWordsToWrite = nw * 2;

    if (mAsBinary) {
      mOutFile.write(reinterpret_cast<const char*>(&numWordsToWrite), sizeof(int));
      mOutFile.write(reinterpret_cast<const char*>(mBufferFloat.data()), nw * sizeof(mBufferFloat[0]));
      mOutFile.write(reinterpret_cast<const char*>(mBufferInt.data()), nw * sizeof(mBufferInt[0]));
    } else {
      mOutFile << numWordsToWrite << "\n";
      for (int i = 0; i < nw; i++) {
        mOutFile << mBufferFloat[i] << " ";
      }
      mOutFile << "\n";
      for (int i = 0; i < nw; i++) {
        mOutFile << mBufferInt[i] << " ";
      }
      mOutFile << "\n";
    }
    wrote = nw * (sizeof(mBufferFloat[0]) + sizeof(mBufferInt[0])) + sizeof(int);
  }
  clear();
  return wrote;
}

//___________________________________________________________________________
/// Initialize for new set of locals, e.g. new track.
void Mille::clear()
{
  mHasSpecial = false;
  mBufferInt.clear();
  mBufferFloat.clear();
  mBufferFloat.push_back(0.f);
  mBufferInt.push_back(0); // position 0 used as error counter
}

} // namespace align
} // namespace o2
