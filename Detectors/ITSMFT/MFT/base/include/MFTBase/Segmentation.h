/// \file Segmentation.h
/// \brief Class for the virtual segmentation of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_SEGMENTATION_H_
#define ALICEO2_MFT_SEGMENTATION_H_

#include "TNamed.h"
#include "TClonesArray.h"

namespace o2 { namespace MFT { class HalfSegmentation; } }

namespace o2 {
namespace MFT {

class Segmentation : public TNamed {

public:
  
  enum { Bottom, Top };

  Segmentation();
  Segmentation(const Char_t *nameGeomFile);
  
  ~Segmentation() override;
  void Clear(const Option_t* /*opt*/) override;

  /// \brief Returns pointer to the segmentation of the half-MFT
  /// \param iHalf Integer : 0 = Bottom; 1 = Top
  /// \return Pointer to a HalfSegmentation
  HalfSegmentation* getHalf(Int_t iHalf) const;

  Int_t getDetElemLocalID(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const;
  
  Bool_t hitToPixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t half, Int_t disk, Int_t ladder, Int_t sensor, Int_t &xPixel, Int_t &yPixel);

private:

  TClonesArray *mHalves; ///< \brief Array of pointer to HalfSegmentation

  ClassDefOverride(Segmentation, 1);
  
};

}
}

#endif

