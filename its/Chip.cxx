//
//  Chip.cpp
//  ALICEO2
//
//  Created by Markus Fasel on 23.07.15.
//  Adapted from AliITSUChip by Massimo Masera
//

#include <TMath.h>

#include "FairLogger.h"

#include "its/Chip.h"
#include "its/Point.h"
#include "its/UpgradeGeometryTGeo.h"

ClassImp(AliceO2::ITS::Chip)

using namespace AliceO2::ITS;

Chip::Chip():
TObject(),
fChipIndex(-1),
fPoints(),
fGeometry(nullptr)
{
  fPoints.SetOwner(kFALSE);
}

Chip::Chip(Int_t chipindex, UpgradeGeometryTGeo *geometry):
TObject(),
fChipIndex(chipindex),
fPoints(),
fGeometry(geometry)
{
  fPoints.SetOwner(kFALSE);
}

Chip::Chip(const Chip &ref):
TObject(ref),
fChipIndex(ref.fChipIndex),
fPoints(ref.fPoints),
fGeometry(ref.fGeometry)
{
}

Chip &Chip::operator=(const Chip &ref){
  TObject::operator=(ref);
  if (this != &ref){
    fGeometry = ref.fGeometry;
    fChipIndex = ref.fChipIndex;
    fPoints = ref.fPoints;
  }
  return *this;
}

Bool_t Chip::operator==(const Chip &other) const {
  return fChipIndex == other.fChipIndex;
}

Bool_t Chip::operator!=(const Chip &other) const {
  return fChipIndex != other.fChipIndex;
}

Bool_t Chip::operator<(const Chip &other) const {
  return fChipIndex < other.fChipIndex;
}

Point *Chip::operator[](Int_t i) const {
  return GetPointAt(i);
}

Chip::~Chip(){
  
}

void Chip::InsertPoint(Point *p){
  if (p->GetDetectorID() != fChipIndex) {
    throw IndexException(fChipIndex, p->GetDetectorID());
  }
  fPoints.AddLast(p);
}

Point *Chip::GetPointAt(Int_t i) const {
  Point * result = nullptr;
  if (i < fPoints.GetEntriesFast()) {
    result = static_cast<Point *>(fPoints.At(i));
  }
  return result;
}

void Chip::Clear(Option_t *opt){
  fPoints.Clear();
}

Bool_t Chip::LineSegmentLocal(Int_t hitindex, Double_t &xstart, Double_t &xpoint, Double_t &ystart, Double_t &ypoint, Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const {
  if (hitindex >= fPoints.GetEntriesFast()) {
    return kFALSE;
  }
  Point *tmp = static_cast<Point *>(fPoints.At(hitindex));
  if (tmp->IsEntering()) {
    return kFALSE;
  }
  Double_t  posglob[3] = { tmp->GetX(), tmp->GetY(), tmp->GetZ()},
  posglobStart[3] = {tmp->GetStartX(), tmp->GetStartY(), tmp->GetStartZ()},
  posloc[3], poslocStart[3];
  memset(posloc, 0, sizeof(Double_t)*3);
  memset(poslocStart, 0, sizeof(Double_t)*3);
  
  // convert to local position
  fGeometry->globalToLocal(fChipIndex, posglob, posloc);
  fGeometry->globalToLocal(fChipIndex, posglobStart, poslocStart);
  
  // Prepare output, hit point relative to starting point
  xstart = poslocStart[0];
  ystart = poslocStart[1];
  zstart = poslocStart[2];
  xpoint = posloc[0] - poslocStart[0];
  ypoint = posloc[1] - poslocStart[1];
  zpoint = posloc[2] - poslocStart[2];
  
  timestart = tmp->GetStartTime();
  eloss = tmp->GetEnergyLoss();
  
  return kTRUE;
}

Bool_t Chip::LineSegmentGlobal(Int_t hitindex, Double_t &xstart, Double_t &xpoint, Double_t &ystart, Double_t &ypoint, Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const {
  if (hitindex >= fPoints.GetEntriesFast()) {
    return kFALSE;
  }
  Point *tmp = static_cast<Point *>(fPoints.At(hitindex));
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
  timestart = tmp->GetStartTime();
  eloss = tmp->GetEnergyLoss();
  
  return kTRUE;
}

Double_t Chip::PathLength(const AliceO2::ITS::Point *p1, const AliceO2::ITS::Point *p2) const {
  Double_t xdiff = p2->GetX() - p1->GetX(),
  ydiff = p2->GetY() - p1->GetY(),
  zdiff = p2->GetZ() - p1->GetZ();
  return TMath::Sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
}

void Chip::MedianHitGlobal(const Point *p1, const Point *p2, Double_t &x, Double_t &y, Double_t &z) const {
  // Get hit positions in global coordinates
  Double_t pos1Glob[3] = {p1->GetX(), p1->GetY(), p1->GetZ()},
  pos2Glob[3] = {p2->GetX(), p2->GetY(), p2->GetZ()}, posMedianLocal[3], posMedianGlobal[3];
  
  // Calculate mean positions
  posMedianLocal[1] = 0.;
  if ((pos1Glob[1] * pos2Glob[1]) < 0.) {
    posMedianLocal[0] = (-pos1Glob[1]/(pos2Glob[1] - pos1Glob[1])) * (pos2Glob[0] - pos1Glob[0]) + pos1Glob[0];
    posMedianLocal[2] = (-pos1Glob[1]/(pos2Glob[1] - pos1Glob[1])) * (pos2Glob[2] - pos1Glob[2]) + pos1Glob[2];
  } else {
    posMedianLocal[0] = 0.5 * (pos1Glob[0] + pos2Glob[0]);
    posMedianLocal[2] = 0.5 * (pos1Glob[2] + pos2Glob[2]);
  }

  // Convert to global coordinates
  fGeometry->localToGlobal(fChipIndex, posMedianLocal, posMedianGlobal);
  x = posMedianGlobal[0];
  y = posMedianGlobal[1];
  z = posMedianGlobal[2];
}

void Chip::MedianHitLocal(const Point *p1, const Point *p2, Double_t &x, Double_t &y, Double_t &z) const {
  // Convert hit positions into local positions inside the chip
  Double_t pos1Glob[3] = {p1->GetX(), p1->GetY(), p1->GetZ()},
  pos2Glob[3] = {p2->GetX(), p2->GetY(), p2->GetZ()}, pos1Loc[3], pos2Loc[3];
  fGeometry->globalToLocal(fChipIndex, pos1Glob, pos1Loc);
  fGeometry->globalToLocal(fChipIndex, pos2Glob, pos2Loc);
  
  // Calculate mean positions
  y = 0.;
  if ((pos1Loc[1] * pos2Loc[1]) < 0.) {
    x = (-pos1Loc[1]/(pos2Loc[1] - pos1Loc[1])) * (pos2Loc[0] - pos1Loc[0]) + pos1Loc[0];
    z = (-pos1Loc[1]/(pos2Loc[1] - pos1Loc[1])) * (pos2Loc[2] - pos1Loc[2]) + pos1Loc[2];
  } else {
    x = 0.5 * (pos1Loc[0] + pos2Loc[0]);
    z = 0.5 * (pos1Loc[2] + pos2Loc[2]);
  }
}