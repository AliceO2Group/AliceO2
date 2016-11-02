/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#include "MFTBase/Constants.h"
#include "MFTSimulation/Geometry.h"
#include "MFTSimulation/GeometryTGeo.h"
#include "MFTSimulation/HalfSegmentation.h"
#include "MFTSimulation/HalfDiskSegmentation.h"
#include "MFTSimulation/LadderSegmentation.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::GeometryTGeo)

TString GeometryTGeo::fgVolumeName   = "MFT";
TString GeometryTGeo::fgHalfDetName  = "MFT_H";
TString GeometryTGeo::fgHalfDiskName = "MFT_D";
TString GeometryTGeo::fgLadderName   = "MFT_L";
TString GeometryTGeo::fgSensorName   = "MFT_S";

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo() : 
TObject(),
fNDisks(0),
fNChips(0),
fNLaddersHalfDisk(0)
{
  // default constructor

  Build();

}

//_____________________________________________________________________________
GeometryTGeo::~GeometryTGeo()
{
  // destructor

  delete [] fNLaddersHalfDisk;

}

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo(const GeometryTGeo& src)
  : TObject(src),
    fNDisks(src.fNDisks),
    fNChips(src.fNChips)
{
  // copy constructor

  fNLaddersHalfDisk = new Int_t[2*src.fNDisks];

  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    for (Int_t iDisk = 0; iDisk < (src.fNDisks); iDisk++) {
      fNLaddersHalfDisk[iHalf*(src.fNDisks)+iDisk] = src.fNLaddersHalfDisk[iHalf*(src.fNDisks)+iDisk];
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
  fNDisks = src.fNDisks;
  fNChips = src.fNChips;

  fNLaddersHalfDisk = new Int_t[2*src.fNDisks];

  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    for (Int_t iDisk = 0; iDisk < (src.fNDisks); iDisk++) {
      fNLaddersHalfDisk[iHalf*(src.fNDisks)+iDisk] = src.fNLaddersHalfDisk[iHalf*(src.fNDisks)+iDisk];
    }
  }

}

//_____________________________________________________________________________
void GeometryTGeo::Build()
{

  fNDisks = Constants::kNDisks;
  fNLaddersHalfDisk = new Int_t[2*fNDisks];

  // extract the total number of sensors (chips)
  Geometry *mftGeo = Geometry::Instance();
  Segmentation *seg = mftGeo->GetSegmentation();
  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation * halfSeg = seg->GetHalf(iHalf);
    for (Int_t iDisk = 0; iDisk < fNDisks; iDisk++) {
      HalfDiskSegmentation* halfDiskSeg = halfSeg->GetHalfDisk(iDisk);
      fNLaddersHalfDisk[iHalf*fNDisks+iDisk] = halfDiskSeg->GetNLadders();
      for (Int_t iLadder = 0; iLadder < halfDiskSeg->GetNLadders(); iLadder++) {
	LadderSegmentation* ladderSeg = halfDiskSeg->GetLadder(iLadder);
	fNChips += ladderSeg->GetNSensors();
      }
    }
  }

}

