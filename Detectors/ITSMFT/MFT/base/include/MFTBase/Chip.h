/// \file Chip.h
/// \brief Class describing geometry of MFT CMOS MAP Chip 
///
/// units are cm and deg
///
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_CHIP_H_
#define ALICEO2_MFT_CHIP_H_

#include "TNamed.h"
#include "TGeoVolume.h"

namespace o2 { namespace MFT { class LadderSegmentation; } }
namespace o2 { namespace MFT { class ChipSegmentation;   } }

namespace o2 {
namespace MFT {

class Chip : public TNamed {
  
public:
  
  Chip();
  Chip(ChipSegmentation *segmentation, const char * ladderName);
  
  ~Chip() override;
  
  TGeoVolume * createVolume();
  void getPosition(LadderSegmentation * ladderSeg, Int_t iChip, Double_t *pos);

private:
  
  ClassDefOverride(Chip, 1);

};

}
}

#endif
