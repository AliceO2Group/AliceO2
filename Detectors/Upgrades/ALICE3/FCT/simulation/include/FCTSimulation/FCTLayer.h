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

/// \file FCTLayer.h
/// \brief Definition of the FCTLayer class

#ifndef ALICEO2_FCT_UPGRADEV3LAYER_H_
#define ALICEO2_FCT_UPGRADEV3LAYER_H_

#include "Rtypes.h"                 // for Double_t, Int_t, Bool_t, etc
#include "FCTSimulation/Detector.h" // for Detector, Detector::Model

class TGeoVolume;

namespace o2
{
namespace fct
{

/// This class defines the Geometry for the FCT Layer TGeo. This is a work class used
/// to study different configurations during the development of the ALICE3 EndCaps
class FCTLayer : public TObject
{
 public:
  // Default constructor
  FCTLayer() = default;

  // Sample layer constructor, for disk and square layers (since they have the same parameters, but a different shape. Differentiate by filling in the type)
  FCTLayer(Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut_SideL, Float_t Layerx2X0, Int_t type);

  /// Copy constructor
  FCTLayer(const FCTLayer&) = default;

  /// Assignment operator
  FCTLayer& operator=(const FCTLayer&) = default;

  /// Default destructor
  ~FCTLayer() override;

  /// getters
  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getSideLength() const { return mSideLength; }
  auto getLayerNumber() const { return mLayerNumber; }
  auto getType() const { return mType; }
  auto getZ() const { return mZ; }
  auto getx2X0() const { return mx2X0; }

  /// Creates the actual Layer and places inside its mother volume
  /// \param motherVolume the TGeoVolume owing the volume structure
  virtual void createLayer(TGeoVolume* motherVolume);

 private:
  Int_t mLayerNumber = -1; ///< Current layer number (converter layers have layer numbers too, but they get their own vector of instances, mBackwardLayers)
  Int_t mType;             ///< Layer type. 0: Disk, 1: square, 2: passive disk converter
  std::string mLayerName;  ///< Current layer name
  Double_t mInnerRadius;   ///< Inner radius of this layer
  Double_t mOuterRadius;   ///< Outer radius of this disk layer
  Double_t mSideLength;    ///< Side length of this square layer
  Double_t mZ;             ///< Z position of the layer
  Double_t mChipThickness; ///< Chip thickness
  Double_t mx2X0;          ///< Layer material budget x/X0

  virtual void createDiskLayer(TGeoVolume* motherVolume);
  virtual void createSquareLayer(TGeoVolume* motherVolume);
  virtual void createConverterLayer(TGeoVolume* motherVolume);

  ClassDefOverride(FCTLayer, 0); // ALICE 3 EndCaps geometry
};
} // namespace fct
} // namespace o2

#endif
