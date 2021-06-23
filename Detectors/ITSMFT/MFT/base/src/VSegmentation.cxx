// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file VSegmentation.cxx
/// \brief Abstract base class for MFT Segmentation description
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "MFTBase/VSegmentation.h"

using namespace o2::mft;

ClassImp(o2::mft::VSegmentation);

//_____________________________________________________________________________
VSegmentation::VSegmentation() : TNamed(), mTransformation(new TGeoCombiTrans())
{
  /// Default constructor
}

//_____________________________________________________________________________
VSegmentation::VSegmentation(const VSegmentation& input) : TNamed(), mTransformation(input.mTransformation)
{
  /// Copy constructor

  SetUniqueID(input.GetUniqueID());
  SetName(input.GetName());
}

//_____________________________________________________________________________
void VSegmentation::setRotationAngles(const Double_t* ang)
{

  /// Set Rotation Angles
  if (!mTransformation) {
    mTransformation = new TGeoCombiTrans();
  }
  auto* rot = new TGeoRotation();
  rot->SetAngles(ang[0], ang[1], ang[2]); // all angles in degrees
  mTransformation->SetRotation(rot);
}
