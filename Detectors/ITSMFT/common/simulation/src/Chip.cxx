// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  Chip.cpp
//  ALICEO2
//
//  Created by Markus Fasel on 23.07.15.
//  Adapted from AliITSUChip by Massimo Masera
//

#include <cstring>
#include <tuple>

#include <TMath.h>
#include <TObjArray.h>
#include <TClonesArray.h>

#include "ITSMFTSimulation/Chip.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSMFTSimulation/DigiParams.h"

using namespace o2::ITSMFT;

ClassImp(o2::ITSMFT::Chip);

//_______________________________________________________________________
Chip::Chip(const DigiParams* par, Int_t chipindex, const TGeoHMatrix *mat) :
  mParams(par),
  mChipIndex(chipindex),
  mMat(mat)
{
}

//_______________________________________________________________________
Chip::Chip(const Chip &ref) = default;

//_______________________________________________________________________
Chip &Chip::operator=(const Chip &ref)
{
  if (this != &ref) {
    mMat = ref.mMat;
    mChipIndex = ref.mChipIndex;
    mHits = ref.Hits;
    mDigits = ref.mDigits;
  }
  return *this;
}

//_______________________________________________________________________
Bool_t Chip::operator==(const Chip &other) const
{
  return mChipIndex == other.mChipIndex;
}

//_______________________________________________________________________
Bool_t Chip::operator!=(const Chip &other) const
{
  return mChipIndex != other.mChipIndex;
}

//_______________________________________________________________________
Bool_t Chip::operator<(const Chip &other) const
{
  return mChipIndex < other.mChipIndex;
}

//_______________________________________________________________________
void Chip::InsertHit(const Point *p)
{
  if (p->GetDetectorID() != mChipIndex) {
    throw IndexException(mChipIndex, p->GetDetectorID());
  }
  mHits.push_back(p);
}

//_______________________________________________________________________
const Point *Chip::GetHitAt(Int_t i) const
{
  if (i < mHits.size()) {
    return mHits[i];
  }
  return nullptr;
}

//_______________________________________________________________________
void Chip::Clear()
{
  ClearHits();
}


//_______________________________________________________________________
Bool_t Chip::LineSegmentLocal(const Point* hit,
			      Double_t &xstart, Double_t &xpoint,
			      Double_t &ystart, Double_t &ypoint,
			      Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const
{
  if (hit->IsEntering()) return kFALSE;

  Double_t posglob[3] = {hit->GetX(), hit->GetY(), hit->GetZ()},
    posglobStart[3] = {hit->GetStartX(), hit->GetStartY(), hit->GetStartZ()},
    posloc[3], poslocStart[3];
  memset(posloc, 0, sizeof(Double_t) * 3);
  memset(poslocStart, 0, sizeof(Double_t) * 3);

  // convert to local position
  mMat->MasterToLocal(posglob, posloc);
  mMat->MasterToLocal(posglobStart, poslocStart);

  // Prepare output, hit point relative to starting point
  xstart = poslocStart[0];
  ystart = poslocStart[1];
  zstart = poslocStart[2];
  xpoint = posloc[0] - poslocStart[0];
  ypoint = posloc[1] - poslocStart[1];
  zpoint = posloc[2] - poslocStart[2];

  timestart = hit->GetTime();
  eloss = hit->GetEnergyLoss();

  return kTRUE;
}


//_______________________________________________________________________
Bool_t Chip::LineSegmentGlobal(const Point* hit, Double_t &xstart, Double_t &xpoint, Double_t &ystart, Double_t &ypoint,
                               Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const
{
  if (hit->IsEntering()) return kFALSE;

  // Fill output fields
  xstart = hit->GetStartX();
  ystart = hit->GetStartY();
  zstart = hit->GetStartZ();
  xpoint = hit->GetX() - xstart;
  ypoint = hit->GetY() - ystart;
  zpoint = hit->GetY() - zstart;
  timestart = hit->GetTime();
  eloss = hit->GetEnergyLoss();

  return kTRUE;
}

//_______________________________________________________________________
Double_t Chip::PathLength(const Hit *p1, const Hit *p2) const
{
  Double_t xdiff = p2->GetX() - p1->GetX(),
    ydiff = p2->GetY() - p1->GetY(),
    zdiff = p2->GetZ() - p1->GetZ();
  return TMath::Sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
}

//_______________________________________________________________________
void Chip::MedianHitGlobal(const Hit *p1, const Hit *p2, Double_t &x, Double_t &y, Double_t &z) const
{
  // Get hit positions in global coordinates
  Double_t pos1Glob[3] = {p1->GetX(), p1->GetY(), p1->GetZ()},
    pos2Glob[3] = {p2->GetX(), p2->GetY(), p2->GetZ()}, posMedianLocal[3], posMedianGlobal[3];

  // Calculate mean positions
  posMedianLocal[1] = 0.;
  if ((pos1Glob[1] * pos2Glob[1]) < 0.) {
    posMedianLocal[0] = (-pos1Glob[1] / (pos2Glob[1] - pos1Glob[1])) * (pos2Glob[0] - pos1Glob[0]) + pos1Glob[0];
    posMedianLocal[2] = (-pos1Glob[1] / (pos2Glob[1] - pos1Glob[1])) * (pos2Glob[2] - pos1Glob[2]) + pos1Glob[2];
  } else {
    posMedianLocal[0] = 0.5 * (pos1Glob[0] + pos2Glob[0]);
    posMedianLocal[2] = 0.5 * (pos1Glob[2] + pos2Glob[2]);
  }

  // Convert to global coordinates
  mMat->LocalToMaster(posMedianLocal, posMedianGlobal);
  x = posMedianGlobal[0];
  y = posMedianGlobal[1];
  z = posMedianGlobal[2];
}

//_______________________________________________________________________
void Chip::MedianHitLocal(const Hit *p1, const Hit *p2, Double_t &x, Double_t &y, Double_t &z) const
{
  // Convert hit positions into local positions inside the chip
  Double_t pos1Glob[3] = {p1->GetX(), p1->GetY(), p1->GetZ()},
    pos2Glob[3] = {p2->GetX(), p2->GetY(), p2->GetZ()}, pos1Loc[3], pos2Loc[3];
  mMat->MasterToLocal(pos1Glob, pos1Loc);
  mMat->MasterToLocal(pos2Glob, pos2Loc);

  // Calculate mean positions
  y = 0.;
  if ((pos1Loc[1] * pos2Loc[1]) < 0.) {
    x = (-pos1Loc[1] / (pos2Loc[1] - pos1Loc[1])) * (pos2Loc[0] - pos1Loc[0]) + pos1Loc[0];
    z = (-pos1Loc[1] / (pos2Loc[1] - pos1Loc[1])) * (pos2Loc[2] - pos1Loc[2]) + pos1Loc[2];
  } else {
    x = 0.5 * (pos1Loc[0] + pos2Loc[0]);
    z = 0.5 * (pos1Loc[2] + pos2Loc[2]);
  }
}

//_______________________________________________________________________
Digit* Chip::addDigit(UInt_t roframe, UShort_t row, UShort_t col, float charge, Label lbl, double timestamp)
{
  auto key = Digit::getOrderingKey(roframe,row,col);
  auto dig = findDigit(key);
  if (dig) {
    dig->addCharge(charge, lbl);
  }
  else {
    auto digIter= mDigits.emplace(std::make_pair
				  (key,Digit(static_cast<UShort_t>(mChipIndex),roframe, row, col, charge, timestamp)));
    auto pair = digIter.first;
    dig = &(pair->second);
    dig->setLabel(0, lbl);
  }
  return dig;
}

//______________________________________________________________________
void Chip::fillOutputContainer(TClonesArray* digits, UInt_t maxFrame)
{
  // transfer digits with RO Frame < maxFrame to the output array
  if (mDigits.empty()) return;
  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  ULong64_t maxKey = Digit::getOrderingKey(maxFrame+1,0,0);
  for (; iter!=mDigits.end(); ++iter) {
    if (iter->first > maxKey) break; // is the digit ROFrame from the key > the max requested frame
    // apply thrshold
    Digit &dig = iter->second;
    //printf("Chip%d Fr:%d Q:%f R:%d C:%d\n",dig.getChipIndex(),dig.getROFrame(),dig.getCharge(), dig.getRow(),dig.getColumn());

    if (dig.getCharge()>mParams->getThreshold() ) {
      new( (*digits)[digits->GetEntriesFast()] ) Digit( dig );
    }
  }

  //  if (iter!=mDigits.end()) iter--;
  mDigits.erase(itBeg, iter);
  
}
