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
  std::vector<std::vector<std::pair<UShort_t,UShort_t>>> mPreClusters;
  std::vector<Int_t> mLabels;
  
  UShort_t mChipID; ///< ID of the chip being processed
  UShort_t mCol;    ///< Column being processed

  static Float_t mPitchX, mPitchZ; ///< Pixel pitch in X and Z (cm)
  static Float_t mX0, mZ0;         ///< Local X and Y coordinates (cm) of the very 1st pixel
};

}
}
#endif /* ALICEO2_ITS_CLUSTERER_H */
