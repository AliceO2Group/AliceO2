// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagFieldFast.cxx
/// \brief Implementation of the fast magnetic field parametrization MagFieldFast
/// \author ruben.shahoyan@cern.ch
//
#include "Field/MagFieldFast.h"
#include <FairLogger.h>
#include <TString.h>
#include <TSystem.h>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace o2::field;
using namespace std;

ClassImp(o2::field::MagFieldFast);

const float MagFieldFast::kSolR2Max[MagFieldFast::kNSolRRanges] = { 80.f * 80.f, 250.f * 250.f, 400.f * 400.f,
                                                                    423.f * 423.f, 500.f * 500.f };

const float MagFieldFast::kSolZMax = 550.0f;

//_______________________________________________________________________
MagFieldFast::MagFieldFast(const string inpFName) : mFactorSol(1.f)
{
  // c-tor
  if (!inpFName.empty() && !LoadData(inpFName)) {
    LOG(FATAL) << "Failed to initialize from " << inpFName << FairLogger::endl;
  }
}

//_______________________________________________________________________
MagFieldFast::MagFieldFast(float factor, int nomField, const string inpFmt) : mFactorSol(factor)
{
  // c-tor
  if (nomField != 2 && nomField != 5) {
    LOG(FATAL) << "No parametrization for nominal field of " << nomField << " kG" << FairLogger::endl;
  }
  TString pth;
  pth.Form(inpFmt.data(), nomField);
  if (!LoadData(pth.Data())) {
    LOG(FATAL) << "Failed to initialize from " << pth.Data() << FairLogger::endl;
  }
}

//_______________________________________________________________________
bool MagFieldFast::LoadData(const string inpFName)
{
  // load field from text file

  std::ifstream in(gSystem->ExpandPathName(inpFName.data()), std::ifstream::in);
  if (in.fail()) {
    LOG(FATAL) << "Failed to open file " << inpFName << FairLogger::endl;
    return false;
  }
  std::string line;
  int valI, component = -1, nParams = 0, header[4] = { -1, -1, -1, -1 }; // iR, iZ, iQuadrant, nVal
  SolParam* curParam = nullptr;

  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue; // empy or comment
    std::stringstream ss(line);
    int cnt = 0;

    if (component < 0) {
      while (cnt < 4 && (ss >> header[cnt++]))
        ;
      if (cnt != 4) {
        LOG(FATAL) << "Wrong header " << line << FairLogger::endl;
        return false;
      }
      curParam = &mSolPar[header[0]][header[1]][header[2]];
    } else {
      while (cnt < header[3] && (ss >> curParam->parBxyz[component][cnt++]))
        ;
      if (cnt != header[3]) {
        LOG(FATAL) << "Wrong data (npar=" << cnt << ") for param " << header[0] << " " << header[1] << " " << header[2]
                   << " " << header[3] << " " << line << FairLogger::endl;
        return false;
      }
    }
    component++;
    if (component > 2) {
      component = -1; // next header expected
      nParams++;
    }
  }
  //
  LOG(INFO) << "Loaded " << nParams << " params from " << inpFName << FairLogger::endl;
  if (nParams != kNSolRRanges * kNSolZRanges * kNQuadrants) {
    LOG(FATAL) << "Was expecting " << kNSolRRanges * kNSolZRanges * kNQuadrants << " params" << FairLogger::endl;
  }
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::Field(const double xyz[3], double bxyz[3]) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz[kX],xyz[kY],xyz[kZ], zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  bxyz[kX] = CalcPol(par->parBxyz[kX], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  bxyz[kY] = CalcPol(par->parBxyz[kY], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  bxyz[kZ] = CalcPol(par->parBxyz[kZ], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::GetBcomp(EDim comp, const double xyz[3], double& b) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz[kX],xyz[kY],xyz[kZ], zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  b = CalcPol(par->parBxyz[comp], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::GetBcomp(EDim comp, const Point3D<float> xyz, double& b) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz.X(),xyz.Y(),xyz.Z(), zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  b = CalcPol(par->parBxyz[comp], xyz.X(),xyz.Y(),xyz.Z()) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::GetBcomp(EDim comp, const Point3D<float> xyz, float& b) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz.X(),xyz.Y(),xyz.Z(), zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  b = CalcPol(par->parBxyz[comp], xyz.X(),xyz.Y(),xyz.Z()) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::GetBcomp(EDim comp, const float xyz[3], float& b) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz[kX],xyz[kY],xyz[kZ], zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  b = CalcPol(par->parBxyz[comp], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::Field(const float xyz[3], float bxyz[3]) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz[kX],xyz[kY],xyz[kZ], zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  bxyz[kX] = CalcPol(par->parBxyz[kX], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  bxyz[kY] = CalcPol(par->parBxyz[kY], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  bxyz[kZ] = CalcPol(par->parBxyz[kZ], xyz[kX], xyz[kY], xyz[kZ]) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::Field(const Point3D<float> xyz, float bxyz[3]) const
{
  // get field
  int zSeg, rSeg, quadrant;
  if (!GetSegment(xyz.X(),xyz.Y(),xyz.Z(), zSeg, rSeg, quadrant)) {
    return false;
  }
  const SolParam* par = &mSolPar[rSeg][zSeg][quadrant];
  bxyz[kX] = CalcPol(par->parBxyz[kX], xyz.X(), xyz.Y(), xyz.Z()) * mFactorSol;
  bxyz[kY] = CalcPol(par->parBxyz[kY], xyz.X(), xyz.Y(), xyz.Z()) * mFactorSol;
  bxyz[kZ] = CalcPol(par->parBxyz[kZ], xyz.X(), xyz.Y(), xyz.Z()) * mFactorSol;
  //
  return true;
}

//_______________________________________________________________________
bool MagFieldFast::GetSegment(float x, float y, float z, int& zSeg, int& rSeg, int& quadrant) const
{
  // get segment of point location
  const float zGridSpaceInv = 1.f / (kSolZMax * 2 / kNSolZRanges);
  zSeg = -1;
  if (z < kSolZMax) {
    if (z > -kSolZMax)
      zSeg = (z + kSolZMax) * zGridSpaceInv; // solenoid params
    else {                                   // need to check dipole params
      return false;
    }
  } else
    return false;
  // R segment
  float xx = x * x, yy = y * y, rr = xx + yy;
  for (rSeg = 0; rSeg < kNSolRRanges; rSeg++)
    if (rr < kSolR2Max[rSeg])
      break;
  if (rSeg == kNSolRRanges)
    return kFALSE;
  quadrant = GetQuadrant(x, y);
  return true;
}
