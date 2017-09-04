// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
