/// \file VSegmentation.cxx
/// \brief Abstract base class for MFT Segmentation description
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "MFTSimulation/VSegmentation.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::VSegmentation);
/// \endcond

//_____________________________________________________________________________
VSegmentation::VSegmentation():
TNamed(),
fTransformation(new TGeoCombiTrans())
{
  /// Default constructor
}

//_____________________________________________________________________________
VSegmentation::VSegmentation(const VSegmentation& input): 
TNamed(),
fTransformation(input.fTransformation)
{
  /// Copy constructor
  
  SetUniqueID(input.GetUniqueID());
  SetName(input.GetName());
  
}

//_____________________________________________________________________________
void VSegmentation::SetRotationAngles(const Double_t *ang)
{

  /// Set Rotation Angles
  if(!fTransformation) fTransformation = new TGeoCombiTrans();
  TGeoRotation *rot = new TGeoRotation();
  rot->SetAngles(ang[0], ang[1], ang[2]); // all angles in degrees
  fTransformation->SetRotation(rot);
  
}



