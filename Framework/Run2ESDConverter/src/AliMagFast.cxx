/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//
// Fast polynomial parametrization of Alice magnetic field, to be used for reconstruction.
// Solenoid part fitted by Shuto Yamasaki from AliMagWrapCheb in the |Z|<260Interface and R<500 cm
// Dipole part: to do
//
// Author: ruben.shahoyan@cern.ch
//
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include <TSystem.h>
#include "AliLog.h"
#include "AliMagFast.h"

const float AliMagFast::fgkSolR2Max[AliMagFast::kNSolRRanges] =
  {80.f*80.f,250.f*250.f,400.f*400.f,423.f*423.f, 500.f*500.f};

const float AliMagFast::fgkSolZMax = 550.0f;

ClassImp(AliMagFast)

AliMagFast::AliMagFast(Float_t factorSol, Float_t factorDip, Int_t nomField,
                       const char* inpFmt, const char* inpFmtDip, const char* symFmtDip) :
  fFactorSol(factorSol), fFactorDip(factorDip)
{
  // c-tor
  if (nomField!=2 && nomField!=5) {
    AliFatalF("No parametrization for nominal field of %d kG",nomField);
  }

  memset(fSolPar,0,sizeof(SolParam_t)*kNSolRRanges*kNSolZRanges*kNQuadrants);

  TString inpFName = Form(inpFmt,nomField);
  TString inpFNameDip = Form(inpFmtDip,nomField);
  TString symDip = Form(symFmtDip,nomField);
  if (!LoadData(inpFName.Data(), inpFNameDip.Data(), symDip.Data())) {
    AliFatalF("Failed to initialize from %s and %s (%s)", inpFName.Data(), inpFNameDip.Data(), symDip.Data());
  }
}

AliMagFast::AliMagFast(const char* inpFName, const char* inpFNameDip, const char* symDip) :
  fFactorSol(1.f), fFactorDip(1.f), fLibNameDip(inpFNameDip)
{
  // c-tor
  memset(fSolPar,0,sizeof(SolParam_t)*kNSolRRanges*kNSolZRanges*kNQuadrants);

  if (!LoadData(inpFName, inpFNameDip, symDip)) {
    AliFatalF("Failed to initialize from %s and %s (%s)", inpFName, inpFNameDip, symDip);
  }
}

//_______________________________________________________________________
AliMagFast::AliMagFast(const AliMagFast &src):
  TObject(src), fFactorSol(src.fFactorSol), fFactorDip(src.fFactorDip), fDipSegments(src.fDipSegments), fDipPar(src.fDipPar), fLibNameDip(src.fLibNameDip)
{
  //AliInfoF("AliMagFast::AliMagFast(const AliMagFast &src) %d",0);
  memcpy(fSolPar,src.fSolPar, kNSolRRanges*kNSolZRanges*kNQuadrants*sizeof(SolParam_t));

}

AliMagFast& AliMagFast::operator=(const AliMagFast& src)
{
  if (this != &src) {
    //AliInfoF("AliMagFast::operator=%d",0);
    fFactorSol = src.fFactorSol;
    fFactorDip = src.fFactorDip;
    fDipSegments = src.fDipSegments;
    fDipPar = src.fDipPar;

    memcpy(fSolPar,src.fSolPar, kNSolRRanges*kNSolZRanges*kNQuadrants*sizeof(SolParam_t));

    fLibNameDip = src.fLibNameDip;
  }
  return *this;
}

AliMagFast::~AliMagFast()
{
}

Bool_t AliMagFast::LoadData(const char* inpFName, const char* inpFNameDip, const char* symDip)
{
  // load field from text file

  std::ifstream in(gSystem->ExpandPathName(inpFName),std::ifstream::in);
  if (in.bad()) {
    AliFatalF("Failed to open input file %s",inpFName);
    return kFALSE;
  }
  std::string line;
  int component = -1, nParams = 0, header[4] = {-1,-1,-1,-1}; // iR, iZ, iQuadrant, nVal
  SolParam_t* curParam = 0; //std::nullptr;

  while (std::getline(in, line)) {

    if (line.empty() || line[0]=='#') continue; // empty or comment
    std::stringstream ss(line);
    int Counter=0;

    if (component<0) {
      while (Counter<4 && (ss>>header[Counter++]));
      if (Counter!=4) {
        AliFatalF("Wrong header: %s",line.c_str());
        return false;
      }
      curParam = &fSolPar[header[0]][header[1]][header[2]];
    }
    else {
      while (Counter<header[3] && (ss>>curParam->mParBxyz[component][Counter++]));
      if (Counter!=header[3]) {
        AliFatalF("Wrong data (npar=%d) for param %d %d %d %d: %s",
                  Counter,header[0],header[1],header[2],header[3],line.c_str());
        return false;
      }
    }
    component++;
    if (component>2) {
      component = -1; // next header expected
      nParams++;
    }
  }
  //
  AliInfoF("loaded %d params for solenoid from %s",nParams,inpFName);
  // load dipole
  const char* dipPath = gSystem->DynamicPathName(inpFNameDip);
  if (!dipPath || gSystem->Load(dipPath) < 0) {
    AliFatalF("Failed to load dipole parameterization %s", inpFNameDip);
    return kFALSE;
  }
  else {
    AliInfoF("loaded library %s for dipole",inpFNameDip);
  }

  TString symSeg = Form("%s_z", symDip);
  TString symPar = Form("%s_params", symDip);
  Func_t dipSeg = gSystem->DynFindSymbol("*", symSeg.Data());
  Func_t dipPar = gSystem->DynFindSymbol("*", symPar.Data());

  if (!dipSeg || !dipPar) {
    gSystem->Unload(inpFNameDip);
    AliFatalF("Failed to find symbols from %s", inpFNameDip);
    return kFALSE;
  }

  fDipSegments = *(SegmentSearch_t*)dipSeg;
  fDipPar = (ChebFormula_t*)dipPar;
  fLibNameDip = TString(inpFNameDip);

  return kTRUE;
}

// get B-field at xyz
Bool_t AliMagFast::Field(const double xyz[3], double bxyz[3]) const
{
  const float fxyz[3]={float(xyz[0]),float(xyz[1]),float(xyz[2])};
  // solenoid
  if (fxyz[2] > -fgkSolZMax) {
    int zSeg,rSeg,quadrant;
    if (!GetSegmentSol(fxyz,zSeg,rSeg,quadrant)) return kFALSE;
    const SolParam_t *par = &fSolPar[rSeg][zSeg][quadrant];
    bxyz[kX] = CalcPol(par->mParBxyz[kX],fxyz[kX],fxyz[kY],fxyz[kZ])*fFactorSol;
    bxyz[kY] = CalcPol(par->mParBxyz[kY],fxyz[kX],fxyz[kY],fxyz[kZ])*fFactorSol;
    bxyz[kZ] = CalcPol(par->mParBxyz[kZ],fxyz[kX],fxyz[kY],fxyz[kZ])*fFactorSol;
  } else {
  // dipole
    UShort_t fid;
    float b[3] = {0.f,0.f,0.f};
    if (!GetSegmentDip(fxyz,fid)) return kFALSE;
    fDipPar[fid].bxyz(fxyz,b);
    bxyz[0]=b[0]*fFactorDip;
    bxyz[1]=b[1]*fFactorDip;
    bxyz[2]=b[2]*fFactorDip;
  }
  //
  return kTRUE;
}

// get Bz at xyz
Bool_t AliMagFast::GetBz(const double xyz[3], double& bz) const
{
  const float fxyz[3]={float(xyz[0]),float(xyz[1]),float(xyz[2])};
  // solenoid
  if (fxyz[2] > -fgkSolZMax) {
    int zSeg,rSeg,quadrant;
    if (!GetSegmentSol(fxyz,zSeg,rSeg,quadrant)) return kFALSE;
    const SolParam_t *par = &fSolPar[rSeg][zSeg][quadrant];
    bz = CalcPol(par->mParBxyz[kZ],fxyz[kX],fxyz[kY],fxyz[kZ])*fFactorSol;
  } else {
  // dipole
    UShort_t fid;
    if (!GetSegmentDip(fxyz,fid)) return kFALSE;
    bz = fDipPar[fid].bz(fxyz)*fFactorDip;
  }
  //
  return kTRUE;
}

// get B-field at xyz
Bool_t AliMagFast::Field(const float xyz[3], float bxyz[3]) const
{
  // solenoid
  if (xyz[2] > -fgkSolZMax) {
    int zSeg,rSeg,quadrant;
    if (!GetSegmentSol(xyz,zSeg,rSeg,quadrant)) return kFALSE;
    const SolParam_t *par = &fSolPar[rSeg][zSeg][quadrant];
    bxyz[kX] = CalcPol(par->mParBxyz[kX],xyz[kX],xyz[kY],xyz[kZ])*fFactorSol;
    bxyz[kY] = CalcPol(par->mParBxyz[kY],xyz[kX],xyz[kY],xyz[kZ])*fFactorSol;
    bxyz[kZ] = CalcPol(par->mParBxyz[kZ],xyz[kX],xyz[kY],xyz[kZ])*fFactorSol;
  } else {
  // dipole
    UShort_t fid;
    if (!GetSegmentDip(xyz,fid)) return kFALSE;
    fDipPar[fid].bxyz(xyz,bxyz);
    bxyz[0] *= fFactorDip;
    bxyz[1] *= fFactorDip;
    bxyz[2] *= fFactorDip;
  }
  //
  return kTRUE;
}

// get Bz at xyz
Bool_t AliMagFast::GetBz(const float xyz[3], float& bz) const
{
  // solenoid
  if (xyz[2] > -fgkSolZMax) {
    int zSeg,rSeg,quadrant;
    if (!GetSegmentSol(xyz,zSeg,rSeg,quadrant)) return kFALSE;
    const SolParam_t *par = &fSolPar[rSeg][zSeg][quadrant];
    bz = CalcPol(par->mParBxyz[kZ],xyz[kX],xyz[kY],xyz[kZ])*fFactorSol;
  } else {
  // dipole
    UShort_t fid;
    if (!GetSegmentDip(xyz,fid)) return kFALSE;
    bz = fDipPar[fid].bz(xyz)*fFactorDip;
  }
  //
  return kTRUE;
}

Bool_t AliMagFast::GetSegmentSol(const float xyz[3], int& zSeg,int &rSeg, int &quadrant) const
{
  // get segment of point location
  const float &x=xyz[kX],&y=xyz[kY],&z=xyz[kZ];
  const float zGridSpaceInv = 1/(fgkSolZMax*2/kNSolZRanges);
  zSeg = -1;
  if (z<fgkSolZMax) {
    if (z>-fgkSolZMax) zSeg = (z+fgkSolZMax)*zGridSpaceInv; // solenoid params
    else { // need to check dipole params
      return kFALSE;
    }
    // check for over/underflow due to the machine precision
    if (zSeg==kNSolZRanges) zSeg = kNSolZRanges-1;
    else if (zSeg==-1) zSeg=0;
  }
  else return kFALSE;
  // R segment
  float xx = x*x, yy = y*y, rr = xx+yy;
  for (rSeg=0;rSeg<kNSolRRanges;rSeg++) if (rr < fgkSolR2Max[rSeg]) break;
  if (rSeg==kNSolRRanges) return kFALSE;
  quadrant = GetQuadrant(x,y);
  return kTRUE;
}
