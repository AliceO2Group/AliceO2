// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideChip.h
/// \brief Creates an ALPIDE chip in simulation
/// \author Mario.Sitta@cern.ch - 24 oct 2017

#ifndef ALICEO2_ITSMFT_ALPIDECHIP_H_
#define ALICEO2_ITSMFT_ALPIDECHIP_H_

#include <TGeoManager.h>   // for gGeoManager
#include "Rtypes.h" // for Int_t, Double_t, Bool_t, UInt_t, etc

class TGeoVolume;
class TGeoManager;
  
namespace o2
{
  
namespace ITSMFT
{

/// AlpideChip class creates a TGeoVolume representing the Alpide Chip
/// Can be used by both ITS and MFT

class AlpideChip
{
 public:

  AlpideChip() = default;
  ~AlpideChip() = default;

  /// Creates the AlpideChip
  /// Returns the chip as a TGeoVolume
  /// \param yc Y chip half lengths
  /// \param ys sensor half thickness
  /// \param chipName sensName default volume names
  /// \param dummy if true creates a dummy air volume (for material budget studies)
  /// \param mgr The GeoManager (used only to get the proper material)
  static TGeoVolume* createChip(Double_t yc, Double_t ys,
                                char const *chipName="AlpideChip", char const *sensName="AlpideSensor",
                                Bool_t dummy=kFALSE, const TGeoManager *mgr=gGeoManager);

  static constexpr Double_t sMetalLayerThick = 15.0*1.0E-4;  ///< Metal layer thickness (um)

  ClassDefNV(AlpideChip, 0) // AlpideChip geometry
};

}
}

#endif
