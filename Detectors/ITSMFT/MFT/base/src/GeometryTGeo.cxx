/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/LadderSegmentation.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::GeometryTGeo)

TString GeometryTGeo::sVolumeName   = "MFT";
TString GeometryTGeo::sHalfDetName  = "MFT_H";
TString GeometryTGeo::sHalfDiskName = "MFT_D";
TString GeometryTGeo::sLadderName   = "MFT_L";
TString GeometryTGeo::sSensorName   = "MFT_S";

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo() : 
TObject(),
mNDisks(0),
mNChips(0),
mNLaddersHalfDisk(nullptr)
{
  // default constructor

  Build();

}

//_____________________________________________________________________________
GeometryTGeo::~GeometryTGeo()
{
  // destructor

  delete [] mNLaddersHalfDisk;

}

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo(const GeometryTGeo& src)
  : TObject(src),
    mNDisks(src.mNDisks),
    mNChips(src.mNChips)
{
  // copy constructor

  mNLaddersHalfDisk = new Int_t[2*src.mNDisks];

  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    for (Int_t iDisk = 0; iDisk < (src.mNDisks); iDisk++) {
      mNLaddersHalfDisk[iHalf*(src.mNDisks)+iDisk] = src.mNLaddersHalfDisk[iHalf*(src.mNDisks)+iDisk];
    }
  }

}

//_____________________________________________________________________________
GeometryTGeo &GeometryTGeo::operator=(const GeometryTGeo &src)
{

  if (this == &src) {
    return *this;
  }

  TObject::operator=(src);
  mNDisks = src.mNDisks;
  mNChips = src.mNChips;

  mNLaddersHalfDisk = new Int_t[2*src.mNDisks];

  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    for (Int_t iDisk = 0; iDisk < (src.mNDisks); iDisk++) {
      mNLaddersHalfDisk[iHalf*(src.mNDisks)+iDisk] = src.mNLaddersHalfDisk[iHalf*(src.mNDisks)+iDisk];
    }
  }

}

//_____________________________________________________________________________
void GeometryTGeo::Build()
{

  mNDisks = Constants::sNDisks;
  mNLaddersHalfDisk = new Int_t[2*mNDisks];

  // extract the total number of sensors (chips)
  Geometry *mftGeo = Geometry::Instance();
  Segmentation *seg = mftGeo->GetSegmentation();
  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation * halfSeg = seg->GetHalf(iHalf);
    for (Int_t iDisk = 0; iDisk < mNDisks; iDisk++) {
      HalfDiskSegmentation* halfDiskSeg = halfSeg->GetHalfDisk(iDisk);
      mNLaddersHalfDisk[iHalf*mNDisks+iDisk] = halfDiskSeg->GetNLadders();
      for (Int_t iLadder = 0; iLadder < halfDiskSeg->GetNLadders(); iLadder++) {
	LadderSegmentation* ladderSeg = halfDiskSeg->GetLadder(iLadder);
	mNChips += ladderSeg->GetNSensors();
      }
    }
  }

}

