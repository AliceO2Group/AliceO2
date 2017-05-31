// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Hit.h
/// \brief Simple hit obtained from points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_HIT_H_
#define ALICEO2_MFT_HIT_H_

#include "FairHit.h"

class TVector3;

namespace o2 {
namespace MFT {

class Hit : public FairHit
{

 public:
  
  Hit();
  Hit(Int_t detID, TVector3& pos, TVector3& dpos, Int_t mcindex);
  
  ~Hit() override;
  
 private:
  
  Hit(const Hit&);
  Hit operator=(const Hit&);

  ClassDefOverride(Hit,1);

};

}
}

#endif
