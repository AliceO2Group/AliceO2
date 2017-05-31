// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_CLUSTERER_H
#define ALICEO2_ITS_CLUSTERER_H

#include <utility>
#include <vector>

#include "Rtypes.h"

class TClonesArray;

namespace o2
{
namespace ITS
{
  class PixelReader;
  class Clusterer
{
 public:
  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  static void setPixelGeometry(Float_t px, Float_t pz, Float_t x0, Float_t z0) {
    mPitchX=px; mPitchZ=pz; mX0=x0; mZ0=z0;
  }
  void process(PixelReader &r, TClonesArray &clusters);

 private:
  enum {kMaxRow=650}; //Anything larger than the real number of rows (512 for ALPIDE)
  void initChip(UShort_t chipID, UShort_t row, UShort_t col, Int_t label);
  void updateChip(UShort_t chipID, UShort_t row, UShort_t col, Int_t label);
  void finishChip(TClonesArray &clusters);

  Int_t mColumn1[kMaxRow+2];
  Int_t mColumn2[kMaxRow+2];
  Int_t *mCurr, *mPrev;
  
  using Pixel = std::pair<UShort_t,UShort_t>;
  using NextIndex = Int_t;
  std::vector< std::pair<NextIndex, Pixel> > mPixels;

  using MCLabel = Int_t;
  using FirstIndex = Int_t;
  std::vector< std::pair<FirstIndex,MCLabel> > mPreClusterHeads;
  
  std::vector<Int_t> mPreClusterIndices;
  
  UShort_t mChipID; ///< ID of the chip being processed
  UShort_t mCol;    ///< Column being processed

  static Float_t mPitchX, mPitchZ; ///< Pixel pitch in X and Z (cm)
  static Float_t mX0, mZ0;         ///< Local X and Y coordinates (cm) of the very 1st pixel
};

}
}
#endif /* ALICEO2_ITS_CLUSTERER_H */
