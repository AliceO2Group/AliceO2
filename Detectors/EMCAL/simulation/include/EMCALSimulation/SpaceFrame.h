// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_SPACEFRAME_H_
#define ALICEO2_EMCAL_SPACEFRAME_H_

#include <FairLogger.h>
#include <Rtypes.h>

namespace o2
{
namespace EMCAL
{
////////////////////////////////////////////////////////////////////////////
///
/// \class  SpaceFrame
/// \brief Space Frame implementation
///
/// EMCAL Space Frame implementation (converted from AliRoot)
///
/// \author Ryan M. Ward, <rmward@calpoly.edu>, Cal Poly
///
////////////////////////////////////////////////////////////////////////////
class SpaceFrame
{
 public:
  ///
  /// Default constructor. Initialize parameters
  SpaceFrame();

  virtual ~SpaceFrame() = default;

  ///
  /// Copy constructor
  SpaceFrame(const SpaceFrame& calFrame);

  SpaceFrame& operator=(const SpaceFrame& /*rvalue*/)
  {
    LOG(FATAL) << "not implemented" << std::endl;
    return *this;
  }

  ///
  /// This method assembles the Geometries and places them into the
  /// Alice master volume
  void CreateGeometry();

 private:
  // space frame parameters from "SINGLE FRAME ASSEMBLY 27D624H.pdf"
  // provided by Lawrence Berkeley Labs, USA
  Int_t mNumCross;          ///< Total number of cross beams including end pieces
  Int_t mNumSubSets;        ///< Total Number of Cross Beam sections in each Half Section
  Double_t mTotalHalfWidth; ///< Half the width of each Half Section
  Double_t mBeginPhi;       ///< Begining Phi of Cal Frame
  Double_t mEndPhi;         ///< Ending Phi of Cal Frame
  Double_t mTotalPhi;       ///< Total Phi range of Cal Frame
  Double_t mBeginRadius;    ///< Begining Radius of Cal Frame
  Double_t mHalfFrameTrans; ///< z-direction translation for each half frame
  // flange and rib dimensions
  Double_t mFlangeHeight; ///< Ending Radius of Flange (Flange is a TGeoTubeSeg)
  Double_t mFlangeWidth;  ///< Thickness of Flange in z-direction
  Double_t mRibHeight;    ///< Ending Radius of Rib
  Double_t mRibWidth;     ///< Thickness of Rib in z-direction
  // cross beam sections - Cross beams come in two sections- top and bottom
  Double_t mCrossBottomWidth;       ///< Width along z direction
  Double_t mCrossTopWidth;          ///< Width along z direction
  Double_t mCrossBottomHeight;      ///< Tangental thickness relative to CalFrame arc
  Double_t mCrossBottomRadThick;    ///< Radial thickness relative to center of geometry
  Double_t mCrossBeamArcLength;     ///< For calulating placement of
  Double_t mCrossBottomStartRadius; ///< Radial position relative to center of geometry
  Double_t mCrossTopHeight;         ///< Tangental thickness relative to the center of geometry
  Double_t mCrossTopRadThick;       ///< Radial thickness relative to CalFrame arc
  Double_t mCrossTopStart;          ///< Radial position relative to center of geometry
  Double_t mEndRadius;              ///< Ending Radius of Mother Volume
  Double_t mEndBeamRadThick;        ///< Radial Thickness of the End Beams
  Double_t mEndBeamBeginRadius;     ///< Starting Radius for End Beams
};
}
}

#endif // ALIEMCALSPACEFRAME_H
