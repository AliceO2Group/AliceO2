/// \file ChipSegmentation.h
/// \brief Chip (sensor) segmentation description
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_CHIPSEGMENTATION_H_
#define ALICEO2_MFT_CHIPSEGMENTATION_H_

#include "MFTBase/VSegmentation.h"

namespace o2 {
namespace MFT {

class ChipSegmentation : public VSegmentation {

public:

  ChipSegmentation();
  ChipSegmentation(UInt_t uniqueID);
  
  ~ChipSegmentation() override = default;
  void Clear(const Option_t* /*opt*/) override {;}
  virtual void print(Option_t* /*option*/);

  /// \brief Transform (x,y) Hit coordinate into Pixel ID on the matrix
  Bool_t hitToPixelID(Double_t xHit, Double_t yHit, Int_t &xPixel, Int_t &yPixel);
  
private:
  
  ClassDefOverride(ChipSegmentation, 1);

};

}
}
	
#endif

