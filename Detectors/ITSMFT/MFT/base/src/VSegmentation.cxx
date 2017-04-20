/// \file VSegmentation.cxx
/// \brief Abstract base class for MFT Segmentation description
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "MFTBase/VSegmentation.h"

using namespace o2::MFT;

ClassImp(o2::MFT::VSegmentation);

//_____________________________________________________________________________
VSegmentation::VSegmentation():
TNamed(),
mTransformation(new TGeoCombiTrans())
{
  /// Default constructor
}

//_____________________________________________________________________________
VSegmentation::VSegmentation(const VSegmentation& input): 
TNamed(),
mTransformation(input.mTransformation)
{
  /// Copy constructor
  
  SetUniqueID(input.GetUniqueID());
  SetName(input.GetName());
  
}

//_____________________________________________________________________________
void VSegmentation::setRotationAngles(const Double_t *ang)
{

  /// Set Rotation Angles
  if(!mTransformation) mTransformation = new TGeoCombiTrans();
  auto *rot = new TGeoRotation();
  rot->SetAngles(ang[0], ang[1], ang[2]); // all angles in degrees
  mTransformation->SetRotation(rot);
  
}



