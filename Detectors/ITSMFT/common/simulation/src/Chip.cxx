//
//  Chip.cpp
//  ALICEO2
//
//  Created by Markus Fasel on 23.07.15.
//  Adapted from AliITSUChip by Massimo Masera
//

#include <cstring>                   // for memset

#include <TMath.h>                    // for Sqrt
#include "TObjArray.h"                // for TObjArray

#include "ITSMFTSimulation/Chip.h"
#include "ITSMFTSimulation/Point.h"

ClassImp(o2::ITSMFT::Chip)

using namespace o2::ITSMFT;

Chip::Chip() :
  TObject(),
  mChipIndex(-1),
  mPoints(),
  mMat(nullptr)
{
}

Chip::Chip(Int_t chipindex, const TGeoHMatrix *mat) :
  TObject(),
  mChipIndex(chipindex),
  mPoints(),
  mMat(mat)
{
}

Chip::Chip(const Chip &ref) = default;

Chip &Chip::operator=(const Chip &ref)
{
  TObject::operator=(ref);
  if (this != &ref) {
    mMat = ref.mMat;
    mChipIndex = ref.mChipIndex;
    mPoints = ref.mPoints;
  }
  return *this;
}

Bool_t Chip::operator==(const Chip &other) const
{
  return mChipIndex == other.mChipIndex;
}

Bool_t Chip::operator!=(const Chip &other) const
{
  return mChipIndex != other.mChipIndex;
}

Bool_t Chip::operator<(const Chip &other) const
{
  return mChipIndex < other.mChipIndex;
}

Chip::~Chip()
= default;

void Chip::InsertPoint(const Point *p)
{
  if (p->GetDetectorID() != mChipIndex) {
    throw IndexException(mChipIndex, p->GetDetectorID());
  }
  mPoints.push_back(p);
}

const Point *Chip::GetPointAt(Int_t i) const
{
  if (i < mPoints.size()) {
    return mPoints[i];
  }
  return nullptr;
}

void Chip::Clear(Option_t *opt)
{
  mPoints.clear();
}

Bool_t Chip::LineSegmentLocal(Int_t hitindex,
Double_t &xstart, Double_t &xpoint,
Double_t &ystart, Double_t &ypoint,
Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const
{
  if (hitindex >= mPoints.size()) {
    return kFALSE;
  }

  const Point *tmp = mPoints[hitindex];
  if (tmp->IsEntering()) {
    return kFALSE;
  }
  Double_t posglob[3] = {tmp->GetX(), tmp->GetY(), tmp->GetZ()},
    posglobStart[3] = {tmp->GetStartX(), tmp->GetStartY(), tmp->GetStartZ()},
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

  timestart = tmp->GetTime();
  eloss = tmp->GetEnergyLoss();

  return kTRUE;
}

Bool_t Chip::LineSegmentGlobal(Int_t hitindex, Double_t &xstart, Double_t &xpoint, Double_t &ystart, Double_t &ypoint,
                               Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const
{
  if (hitindex >= mPoints.size()) {
    return kFALSE;
  }
  const Point *tmp = mPoints[hitindex];
  if (tmp->IsEntering()) {
    return kFALSE;
  }

  // Fill output fields
  xstart = tmp->GetStartX();
  ystart = tmp->GetStartY();
  zstart = tmp->GetStartZ();
  xpoint = tmp->GetX() - xstart;
  ypoint = tmp->GetY() - ystart;
  zpoint = tmp->GetY() - zstart;
  timestart = tmp->GetTime();
  eloss = tmp->GetEnergyLoss();

  return kTRUE;
}

Double_t Chip::PathLength(const Point *p1, const Point *p2) const
{
  Double_t xdiff = p2->GetX() - p1->GetX(),
    ydiff = p2->GetY() - p1->GetY(),
    zdiff = p2->GetZ() - p1->GetZ();
  return TMath::Sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
}

void Chip::MedianHitGlobal(const Point *p1, const Point *p2, Double_t &x, Double_t &y, Double_t &z) const
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

void Chip::MedianHitLocal(const Point *p1, const Point *p2, Double_t &x, Double_t &y, Double_t &z) const
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
