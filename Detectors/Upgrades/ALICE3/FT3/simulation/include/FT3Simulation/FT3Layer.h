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

/// \file FT3Layer.h
/// \brief Definition of the FT3Layer class

#ifndef ALICEO2_FT3_UPGRADEV3LAYER_H_
#define ALICEO2_FT3_UPGRADEV3LAYER_H_

#include <TGeoManager.h>            // for gGeoManager
#include "Rtypes.h"                 // for Double_t, Int_t, Bool_t, etc
#include "FT3Simulation/Detector.h" // for Detector, Detector::Model

class TGeoVolume;

namespace o2
{
namespace ft3
{

/// This class defines the Geometry for the FT3 Layer TGeo. This is a work class used
/// to study different configurations during the development of the ALICE3 EndCaps
class FT3Layer : public TObject
{
 public:
  // Default constructor
  FT3Layer() = default;

  // Sample layer constructor
  FT3Layer(Int_t layerDirection, Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut, Float_t Layerx2X0);

  /// Copy constructor
  FT3Layer(const FT3Layer&) = default;

  /// Assignment operator
  FT3Layer& operator=(const FT3Layer&) = default;

  /// Default destructor
  ~FT3Layer() override;

  /// getters
  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getDirection() const { return mDirection; }
  auto getZ() const { return mZ; }
  auto getx2X0() const { return mx2X0; }

  /// Creates the actual Layer and places inside its mother volume
  /// \param motherVolume the TGeoVolume owing the volume structure
  virtual void createLayer(TGeoVolume* motherVolume);

 private:
  Int_t mLayerNumber = -1; ///< Current layer number
  Int_t mDirection;        ///< Layer direction 0=Forward 1 = Backward
  std::string mLayerName;  ///< Current layer name
  Double_t mInnerRadius;   ///< Inner radius of this layer
  Double_t mOuterRadius;   ///< Outer radius of this layer
  Double_t mZ;             ///< Z position of the layer
  Double_t mChipThickness; ///< Chip thickness
  Double_t mx2X0;          ///< Layer material budget x/X0

  ClassDefOverride(FT3Layer, 0); // ALICE 3 EndCaps geometry
};
} // namespace ft3
} // namespace o2

#endif
