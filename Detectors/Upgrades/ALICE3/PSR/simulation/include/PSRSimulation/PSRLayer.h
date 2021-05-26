// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PSRLayer.h
/// \brief Definition of the PreshowerLayer class

#ifndef ALICEO2_PSR_UPGRADEV3LAYER_H_
#define ALICEO2_PSR_UPGRADEV3LAYER_H_

#include <TGeoManager.h>            // for gGeoManager
#include "Rtypes.h"                 // for Double_t, Int_t, Bool_t, etc
#include "PSRSimulation/Detector.h" // for Detector, Detector::Model

class TGeoVolume;

namespace o2
{
namespace psr
{

/// This class defines the Geometry for the Preshower Layer TGeo. This is a work class used
/// to study different configurations during the development of the ALICE3 EndCaps
class PSRLayer : public TObject
{
 public:
  // Default constructor
  PSRLayer() = default;

  // Sample layer constructor (deleted option: Int_t layerDirection)
  PSRLayer(Int_t layerDirection, Int_t layerNumber, std::string layerName, Float_t z, Float_t rIn, Float_t rOut, Float_t sensorThickness, Float_t Layerx2X0);

  /// Copy constructor
  PSRLayer(const PSRLayer&) = default;

  /// Assignment operator
  PSRLayer& operator=(const PSRLayer&) = default;

  /// Default destructor
  ~PSRLayer() override;

  /// Creates the actual Layer and places inside its mother volume
  /// \param motherVolume the TGeoVolume owing the volume structure
  virtual void createLayer(TGeoVolume* motherVolume);

 private:
  Int_t mLayerNumber = -1;   ///< Current layer number
  Int_t mDirection;          ///< Layer direction 0=Forward 1 = Backward
  std::string mLayerName;    ///< Current layer name
  Double_t mInnerRadius;     ///< Inner radius of this layer
  Double_t mOuterRadius;     ///< Outer radius of this layer
  Double_t mZ;               ///< Z position of the layer
  Double_t mSensorThickness; ///< Sensor thickness
  Double_t mChipThickness;   ///< Chip thickness
  Double_t mx2X0;            ///< Layer material budget x/X0
  //Double_t mThickness;       ///< Thickness of the Pb layer
  ClassDefOverride(PSRLayer, 0); // ALICE 3 EndCaps geometry
};
} // namespace preshower(psr)
} // namespace o2

#endif
