// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSSimulation/GeometryParams.h"
#include "TMath.h"

using namespace o2::phos;

ClassImp(GeometryParams);

GeometryParams* GeometryParams::sGeomParam = nullptr;

//____________________________________________________________________________
GeometryParams::GeometryParams(const std::string_view name)
  : // Set zeros to the variables: most of them should be calculated
    // and it is more clear to set them in the text
    mNModules(4),
    mAngle(0.),
    mIPtoUpperCPVsurface(0.),
    mCrystalShift(0.),
    mCryCellShift(0.),
    mAirGapLed(0.),
    mStripWallWidthOut(0.),
    mStripWallWidthIn(0.),
    mTyvecThickness(0.),
    mInnerThermoWidthX(0.f),
    mInnerThermoWidthY(0.f),
    mInnerThermoWidthZ(0.f),
    mAirGapWidthX(0.f),
    mAirGapWidthY(0.f),
    mAirGapWidthZ(0.f),
    mCoolerWidthX(0.f),
    mCoolerWidthY(0.f),
    mCoolerWidthZ(0.f),
    mAlCoverThickness(0.f),
    mOuterThermoWidthXUp(0.f),
    mOuterThermoWidthXLow(0.f),
    mOuterThermoWidthY(0.f),
    mOuterThermoWidthZ(0.f),
    mAlFrontCoverX(0.f),
    mAlFrontCoverZ(0.f),
    mFiberGlassSup2X(0.f),
    mFiberGlassSup1X(0.f),
    mFrameHeight(0.f),
    mFrameThickness(0.f),
    mAirSpaceFeeX(0.f),
    mAirSpaceFeeZ(0.f),
    mAirSpaceFeeY(0.f),
    mWarmUpperThickness(0.f),
    mWarmBottomThickness(0.f),
    mWarmAlCoverWidthX(0.f),
    mWarmAlCoverWidthY(0.f),
    mWarmAlCoverWidthZ(0.f),
    mFiberGlassSup1Y(0.f),
    mFiberGlassSup2Y(0.f),
    mTSupportDist(0.f),
    mTSupport1Thickness(0.f),
    mTSupport2Thickness(0.f),
    mTSupport1Width(0.f),
    mTSupport2Width(0.f),
    mIPtoOuterCoverDistance(0.f),
    mIPtoCrystalSurface(0.f),
    mSupportPlateThickness(0.f),
    mNCellsXInStrip(0),
    mNCellsZInStrip(0),
    mNStripX(0),
    mNStripZ(0),
    mNTSupports(0),
    mNPhi(0),
    mNz(0),
    mNumberOfCPVLayers(0),
    mNumberOfCPVPadsPhi(0),
    mNumberOfCPVPadsZ(0),
    mCPVPadSizePhi(0.),
    mCPVPadSizeZ(0.),
    mNumberOfCPVChipsPhi(0),
    mNumberOfCPVChipsZ(0),
    mCPVGasThickness(0.),
    mCPVTextoliteThickness(0.),
    mCPVCuNiFoilThickness(0.),
    mDistanceBetwRails(0.),
    mRailsDistanceFromIP(0.),
    mCradleWallThickness(0.)
{
  // Initializes the EMC parameters
  // Coordinate system chosen: x across beam, z along beam, y out of beam.
  // Reference point for all volumes incide module is
  // center of module in x,z on the upper surface of support beam

  // Distance from IP to surface of the crystals
  mIPtoCrystalSurface = 460.0;

  // CRYSTAL

  mCrystalHalfSize[0] = 2.2 / 2; // Half-Sizes of crystall
  mCrystalHalfSize[1] = 18.0 / 2;
  mCrystalHalfSize[2] = 2.2 / 2;

  // APD + preamplifier

  // fPinDiodeSize[0] = 1.71 ;   //Values of ame PIN diode
  // fPinDiodeSize[1] = 0.0280 ; // OHO 0.0280 is the depth of active layer
  // fPinDiodeSize[2] = 1.61 ;

  mPinDiodeHalfSize[0] = 0.5000 / 2; // APD 5 mm side
  mPinDiodeHalfSize[1] = 0.0100 / 2; // APD bulk thickness
  mPinDiodeHalfSize[2] = 0.5000 / 2; // APD 5 mm side

  mPreampHalfSize[0] = 1.5 / 2; // Preamplifier
  mPreampHalfSize[1] = 0.5 / 2;
  mPreampHalfSize[2] = 1.5 / 2;

  // Strip unit (8x2 crystals)

  mNCellsXInStrip = 8; // Number of crystals in strip unit along x-axis
  mNCellsZInStrip = 2; // Number of crystals in strip unit along z-axis
  mNStripX = 8;        // Number of strip units across along x-axis
  mNStripZ = 28;       // Number of strips along z-axis

  mStripWallWidthOut = 0.01; // Side to another strip
  mStripWallWidthIn = 0.02;  // Side betveen crystals in one strip

  mTyvecThickness = 0.0175; // Thickness of the tyvec

  mAirGapLed =
    1.5 - 2 * mPreampHalfSize[1] - 2 * mPinDiodeHalfSize[1]; // Air gap before crystalls for LED system
                                                             // Note, that Cell in Strip 1.5 longer then crystall

  //---Now calculate thechnical sizes for GEANT implementation

  mWrappedHalfSize[0] = (2 * mTyvecThickness + 2 * mCrystalHalfSize[0]) / 2; // This will be size of crystall
  mWrappedHalfSize[1] = mCrystalHalfSize[1];                                 // wrapped into tyvec
  mWrappedHalfSize[2] = (2 * mTyvecThickness + 2 * mCrystalHalfSize[2]) / 2; //

  mAirCellHalfSize[0] = mWrappedHalfSize[0] + 0.01;
  mAirCellHalfSize[1] =
    (mAirGapLed + 2 * mPreampHalfSize[1] + 2 * mPinDiodeHalfSize[1] + 2 * mWrappedHalfSize[1]) / 2; // in strip
  mAirCellHalfSize[2] = mWrappedHalfSize[2] + 0.01;

  //  fSupportPlateHalfSize[0] = ( (fNCellsXInStrip-1)*fStripWallWidthIn + 2*fStripWallWidthOut +
  //			       fNCellsXInStrip * (2*fTyvecThickness + 2*fCrystalHalfSize[0]) )/2 ;
  mSupportPlateHalfSize[0] = 18.04 / 2;
  mSupportPlateHalfSize[1] = 6.0 / 2;
  //  fSupportPlateHalfSize[2] = ( (fNCellsZInStrip-1)*fStripWallWidthIn + 2*fStripWallWidthOut +
  //			       fNCellsZInStrip * (2*fTyvecThickness + 2*fCrystalHalfSize[2]) )/2;
  mSupportPlateHalfSize[2] = 4.51 / 2;
  mSupportPlateThickness = 0.3;
  mSupportPlateInHalfSize[0] = mSupportPlateHalfSize[0];                          // Half-sizes of the air
  mSupportPlateInHalfSize[1] = mSupportPlateHalfSize[1] - mSupportPlateThickness; // box in the support plate
  mSupportPlateInHalfSize[2] = mSupportPlateHalfSize[2] - mSupportPlateThickness / 2;

  mStripHalfSize[0] = mSupportPlateHalfSize[0];
  mStripHalfSize[1] = (2 * mSupportPlateHalfSize[1] + 2 * mAirCellHalfSize[1]) / 2;
  mStripHalfSize[2] = mSupportPlateHalfSize[2];

  // ------- Inner hermoinsulation ---------------
  mInnerThermoWidthX = 2.0; // Width of the innerthermoinsulation across the beam
  mInnerThermoWidthY = 2.0; // Width of the upper cover of innerthermoinsulation
  mInnerThermoWidthZ = 2.0; // Width of the innerthermoinsulation along the beam

  mInnerThermoHalfSize[0] = (2 * mStripHalfSize[0] * mNStripX + 2 * mInnerThermoWidthX) / 2;
  mInnerThermoHalfSize[1] = (2 * mStripHalfSize[1] + mInnerThermoWidthY) / 2;
  mInnerThermoHalfSize[2] = (2 * mStripHalfSize[2] * mNStripZ + 2 * mInnerThermoWidthZ) / 2;

  // ------- Air gap between inner thermoinsulation and passive coller ---------

  mAirGapWidthX = 0.2; // Width of the air gap across the beam
  mAirGapWidthY = 0.2; // Width of the upper air gap
  mAirGapWidthZ = 0.2; // Width of the air gap along the beam

  mAirGapHalfSize[0] = (2 * mInnerThermoHalfSize[0] + 2 * mAirGapWidthX) / 2;
  mAirGapHalfSize[1] = (2 * mInnerThermoHalfSize[1] + mAirGapWidthY) / 2;
  mAirGapHalfSize[2] = (2 * mInnerThermoHalfSize[2] + 2 * mAirGapWidthZ) / 2;

  // ------- Passive Cooler ------------------------

  mCoolerWidthX = 2.0; // Width of the passive coller across the beam
  mCoolerWidthY = 0.3; // Width of the upper cover of cooler
  mCoolerWidthZ = 2.0; // Width of the passive cooler along the beam

  mCoolerHalfSize[0] = (2 * mAirGapHalfSize[0] + 2 * mCoolerWidthX) / 2;
  mCoolerHalfSize[1] = (2 * mAirGapHalfSize[1] + mCoolerWidthY) / 2;
  mCoolerHalfSize[2] = (2 * mAirGapHalfSize[2] + 2 * mCoolerWidthZ) / 2;

  // ------- Outer thermoinsulation and Al cover -------------------------------

  mAlCoverThickness = 0.1; // Thickness of the Al cover of the module

  mOuterThermoWidthXUp = 156.0 - mAlCoverThickness;
  // width of the upper surface of the PHOS module accross the beam
  mOuterThermoWidthY = 6.0; // with of the upper cover of outer thermoinsulation
  mOuterThermoWidthZ = 6.0; // width of the thermoinsulation along the beam

  mAlFrontCoverX = 6.0; // Width of Al strip around fiberglass window: across
  mAlFrontCoverZ = 6.0; // and along the beam

  // Calculate distance from IP to upper cover
  mIPtoOuterCoverDistance = mIPtoCrystalSurface - mAirGapLed - mInnerThermoWidthY - mAirGapWidthY - mCoolerWidthY -
                            mOuterThermoWidthY - mAlCoverThickness;

  Float_t tanA = mOuterThermoWidthXUp / (2. * mIPtoOuterCoverDistance);
  // tan(a) where A = angle between IP to center and IP to side across beam

  mOuterThermoWidthXLow =
    mOuterThermoWidthXUp + 2 * (2 * mCoolerHalfSize[1] + mOuterThermoWidthY) * tanA - mAlCoverThickness;
  // width of the lower surface of the COOL section accross the beam

  mOuterThermoParams[0] = mOuterThermoWidthXUp / 2;  // half-length along x at the z surface positioned at -DZ;
  mOuterThermoParams[1] = mOuterThermoWidthXLow / 2; // half-length along x at the z surface positioned at +DZ;
  mOuterThermoParams[2] = (2 * mCoolerHalfSize[2] + 2 * mOuterThermoWidthZ) / 2;
  // `half-length along the y-axis' in out case this is z axis
  mOuterThermoParams[3] = (2 * mCoolerHalfSize[1] + mOuterThermoWidthY) / 2;
  // `half-length along the z-axis' in our case this is y axis

  mAlCoverParams[0] = mOuterThermoParams[0] + mAlCoverThickness;
  mAlCoverParams[1] = mOuterThermoParams[1] + mAlCoverThickness;
  mAlCoverParams[2] = mOuterThermoParams[2] + mAlCoverThickness;
  mAlCoverParams[3] = mOuterThermoParams[3] + mAlCoverThickness / 2;

  mFiberGlassHalfSize[0] = mAlCoverParams[0] - mAlFrontCoverX;
  mFiberGlassHalfSize[1] = mAlCoverParams[2] - mAlFrontCoverZ; // Note, here other ref. system
  mFiberGlassHalfSize[2] = mAlCoverThickness / 2;

  //============Now warm section======================
  // Al Cover
  mWarmAlCoverWidthX = 2 * mAlCoverParams[1]; // Across beam
  mWarmAlCoverWidthY = 159.0;                 // along beam

  // T-support
  mTSupport1Thickness = 3.5;
  mTSupport2Thickness = 5.0;
  mTSupport1Width = 10.6;
  mTSupport2Width = 3.1;
  mNTSupports = mNStripX + 1;
  mTSupportDist = 7.48;

  // Air space for FEE
  mAirSpaceFeeX = 148.6; // Across beam
  mAirSpaceFeeY = 135.0; // along beam
  mAirSpaceFeeZ = 19.0;  // out of beam

  // thermoinsulation
  mWarmBottomThickness = 4.0;
  mWarmUpperThickness = 4.0;

  // Frame
  mFrameThickness = 5.0;
  mFrameHeight = 15.0;

  // Fiberglass support
  mFiberGlassSup1X = 6.0;
  mFiberGlassSup1Y = 3.9 + mWarmUpperThickness;

  mFiberGlassSup2X = 3.0;
  mFiberGlassSup2Y = mFrameHeight;

  // Now calculate Half-sizes

  mWarmAlCoverWidthZ =
    mAirSpaceFeeZ + mWarmBottomThickness + mWarmUpperThickness + mTSupport1Thickness + mTSupport2Thickness;

  mWarmAlCoverHalfSize[0] = mWarmAlCoverWidthX / 2;
  mWarmAlCoverHalfSize[1] = mWarmAlCoverWidthY / 2;
  mWarmAlCoverHalfSize[2] = mWarmAlCoverWidthZ / 2;

  mWarmThermoHalfSize[0] = mWarmAlCoverHalfSize[0] - mAlCoverThickness;
  mWarmThermoHalfSize[1] = mWarmAlCoverHalfSize[1] - mAlCoverThickness;
  mWarmThermoHalfSize[2] = mWarmAlCoverHalfSize[2] - mAlCoverThickness / 2;

  // T-support
  mTSupport1HalfSize[0] = mTSupport1Width / 2;                        // Across beam
  mTSupport1HalfSize[1] = (mAirSpaceFeeY + 2 * mFiberGlassSup1X) / 2; // along beam
  mTSupport1HalfSize[2] = mTSupport1Thickness / 2;                    // out of beam

  mTSupport2HalfSize[0] = mTSupport2Width / 2;     // Across beam
  mTSupport2HalfSize[1] = mTSupport1HalfSize[1];   // along beam
  mTSupport2HalfSize[2] = mTSupport2Thickness / 2; // out of beam

  // cables
  mTCables1HalfSize[0] =
    (2 * mTSupport1HalfSize[0] * mNTSupports + (mNTSupports - 1) * mTSupportDist) / 2; // Across beam
  mTCables1HalfSize[1] = mTSupport1HalfSize[1];                                        // along beam
  mTCables1HalfSize[2] = mTSupport1HalfSize[2];                                        // out of beam

  mTCables2HalfSize[0] = mTCables1HalfSize[0];  // Across beam
  mTCables2HalfSize[1] = mTSupport2HalfSize[1]; // along beam
  mTCables2HalfSize[2] = mTSupport2HalfSize[2]; // out of beam

  // frame: we define two frames along beam ...Z and across beam ...X
  mFrameXHalfSize[0] = (mAirSpaceFeeX + 2 * mFiberGlassSup2X + 2 * mFrameThickness) / 2;
  mFrameXHalfSize[1] = mFrameThickness / 2;
  mFrameXHalfSize[2] = mFrameHeight / 2;

  mFrameXPosition[0] = 0;
  mFrameXPosition[1] = mAirSpaceFeeY / 2 + mFiberGlassSup2X + mFrameXHalfSize[1];
  mFrameXPosition[2] = mWarmThermoHalfSize[2] - mFrameHeight / 2 - mWarmBottomThickness;

  mFrameZHalfSize[0] = mFrameThickness / 2;
  mFrameZHalfSize[1] = (mAirSpaceFeeY + 2 * mFiberGlassSup2X) / 2;
  mFrameZHalfSize[2] = mFrameHeight / 2;

  mFrameZPosition[0] = mAirSpaceFeeX / 2 + mFiberGlassSup2X + mFrameZHalfSize[0];
  mFrameZPosition[1] = 0;
  mFrameZPosition[2] = mWarmThermoHalfSize[2] - mFrameHeight / 2 - mWarmBottomThickness;

  // Fiberglass support define 4 fiber glass supports 2 along Z  and 2 along X

  mFGupXHalfSize[0] = mFrameXHalfSize[0];
  mFGupXHalfSize[1] = mFiberGlassSup1X / 2;
  mFGupXHalfSize[2] = mFiberGlassSup1Y / 2;

  mFGupXPosition[0] = 0;
  mFGupXPosition[1] = mAirSpaceFeeY / 2 + mFGupXHalfSize[1];
  mFGupXPosition[2] = mWarmThermoHalfSize[2] - mFrameHeight - mWarmBottomThickness - mFGupXHalfSize[2];

  mFGupZHalfSize[0] = mFiberGlassSup1X / 2;
  mFGupZHalfSize[1] = mAirSpaceFeeY / 2;
  mFGupZHalfSize[2] = mFiberGlassSup1Y / 2;

  mFGupZPosition[0] = mAirSpaceFeeX / 2 + mFGupZHalfSize[0];
  mFGupZPosition[1] = 0;
  mFGupZPosition[2] = mWarmThermoHalfSize[2] - mFrameHeight - mWarmBottomThickness - mFGupXHalfSize[2];

  mFGlowXHalfSize[0] = mFrameXHalfSize[0] - 2 * mFrameZHalfSize[0];
  mFGlowXHalfSize[1] = mFiberGlassSup2X / 2;
  mFGlowXHalfSize[2] = mFrameXHalfSize[2];

  mFGlowXPosition[0] = 0;
  mFGlowXPosition[1] = mAirSpaceFeeY / 2 + mFGlowXHalfSize[1];
  mFGlowXPosition[2] = mWarmThermoHalfSize[2] - mWarmBottomThickness - mFGlowXHalfSize[2];

  mFGlowZHalfSize[0] = mFiberGlassSup2X / 2;
  mFGlowZHalfSize[1] = mAirSpaceFeeY / 2;
  mFGlowZHalfSize[2] = mFrameZHalfSize[2];

  mFGlowZPosition[0] = mAirSpaceFeeX / 2 + mFGlowZHalfSize[0];
  mFGlowZPosition[1] = 0;
  mFGlowZPosition[2] = mWarmThermoHalfSize[2] - mWarmBottomThickness - mFGlowXHalfSize[2];

  // --- Air Gap for FEE ----

  mFEEAirHalfSize[0] = mAirSpaceFeeX / 2;
  mFEEAirHalfSize[1] = mAirSpaceFeeY / 2;
  mFEEAirHalfSize[2] = mAirSpaceFeeZ / 2;

  mFEEAirPosition[0] = 0;
  mFEEAirPosition[1] = 0;
  mFEEAirPosition[2] = mWarmThermoHalfSize[2] - mWarmBottomThickness - mFEEAirHalfSize[2];

  // --- Calculate the overol dimentions of the EMC module

  mEMCParams[3] = mAlCoverParams[3] + mWarmAlCoverHalfSize[2]; // Size out of beam
  mEMCParams[0] = mAlCoverParams[0];                           // Upper size across the beam
  mEMCParams[1] = (mAlCoverParams[1] - mAlCoverParams[0]) * mEMCParams[3] / mAlCoverParams[3] +
                  mAlCoverParams[0];       // Lower size across the beam
  mEMCParams[2] = mWarmAlCoverHalfSize[1]; // Size along the beam

  mNPhi = mNStripX * mNCellsXInStrip; // number of crystals across the beam
  mNz = mNStripZ * mNCellsZInStrip;   // number of crystals along the beam

  // CPV Geometry
  mCPVFrameSize[0] = 2.5;
  mCPVFrameSize[1] = 5.1;
  mCPVFrameSize[2] = 2.5;
  mGassiplexChipSize[0] = 4.2;
  mGassiplexChipSize[1] = 0.1;
  mGassiplexChipSize[2] = 6.0;
  mFTPosition[0] = 0.7;
  mFTPosition[1] = 2.2;
  mFTPosition[2] = 3.6;
  mFTPosition[3] = 5.1;

  mCPVActiveSize[0] = mNumberOfCPVPadsPhi * mCPVPadSizePhi;
  mCPVActiveSize[1] = mNumberOfCPVPadsZ * mCPVPadSizeZ;
  mCPVBoxSize[0] = mCPVActiveSize[0] + 2 * mCPVFrameSize[0];
  mCPVBoxSize[1] = mCPVFrameSize[1] * mNumberOfCPVLayers + 0.1;
  mCPVBoxSize[2] = mCPVActiveSize[1] + 2 * mCPVFrameSize[2];

  mPHOSParams[0] =
    TMath::Max((Double_t)mCPVBoxSize[0] / 2.,
               (Double_t)(mEMCParams[0] - (mEMCParams[1] - mEMCParams[0]) * mCPVBoxSize[1] / 2 / mEMCParams[3]));
  mPHOSParams[1] = mEMCParams[1];
  mPHOSParams[2] = TMath::Max((Double_t)mEMCParams[2], (Double_t)mCPVBoxSize[2] / 2.);
  mPHOSParams[3] = mEMCParams[3] + mCPVBoxSize[1] / 2.;

  mIPtoUpperCPVsurface = mIPtoOuterCoverDistance - mCPVBoxSize[1];

  // calculate offset to crystal surface
  mCrystalShift = -mInnerThermoHalfSize[1] + mStripHalfSize[1] + mSupportPlateHalfSize[1] + mCrystalHalfSize[1] -
                  mAirGapLed / 2. + mPinDiodeHalfSize[1] + mPreampHalfSize[1];
  mCryCellShift = mCrystalHalfSize[1] - (mAirGapLed - 2 * mPinDiodeHalfSize[1] - 2 * mPreampHalfSize[1]) / 2;

  Double_t const kRADDEG = 180.0 / TMath::Pi();
  mAngle = 20;
  for (Int_t i = 1; i <= mNModules; i++) {
    Float_t angle = mAngle * (i - 3);
    mPHOSAngle[i - 1] = -angle;
  }

  Float_t r = mIPtoOuterCoverDistance + mPHOSParams[3] - mCPVBoxSize[1];
  for (Int_t iModule = 0; iModule < mNModules; iModule++) {
    mModuleCenter[iModule][0] = r * TMath::Sin(mPHOSAngle[iModule] / kRADDEG);
    mModuleCenter[iModule][1] = -r * TMath::Cos(mPHOSAngle[iModule] / kRADDEG);
    mModuleCenter[iModule][2] = 0.;

    mModuleAngle[iModule][0][0] = 90;
    mModuleAngle[iModule][0][1] = mPHOSAngle[iModule];
    mModuleAngle[iModule][1][0] = 0;
    mModuleAngle[iModule][1][1] = 0;
    mModuleAngle[iModule][2][0] = 90;
    mModuleAngle[iModule][2][1] = 270 + mPHOSAngle[iModule];
  }

  printf("mNModules=%d, modCenter=(%f,%f,%f) \n", mNModules, mModuleCenter[2][0], mModuleCenter[2][1],
         mModuleCenter[2][2]);

  // Support geometry
  mRailLength = 1200.0;
  mDistanceBetwRails = 420.0;
  mRailsDistanceFromIP = 590.;
  mCradleWallThickness = 2.0;

  mRailPart1[0] = 28.0;
  mRailPart1[1] = 3.0;
  mRailPart1[2] = mRailLength;

  mRailPart2[0] = 1.5;
  mRailPart2[1] = 34.0;
  mRailPart2[2] = mRailLength;

  mRailPart3[0] = 6.0;
  mRailPart3[1] = 5.0;
  mRailPart3[2] = mRailLength;

  mRailPos[0] = 0.;
  mRailPos[1] = 0.;
  mRailPos[2] = 0.;

  mRailOuterSize[0] = mRailPart1[0];
  mRailOuterSize[1] = mRailPart1[1] * 2 + mRailPart2[1] + mRailPart3[1];
  mRailOuterSize[2] = mRailLength;

  mRailRoadSize[0] = mDistanceBetwRails + mRailOuterSize[0];
  mRailRoadSize[1] = mRailOuterSize[1];
  mRailRoadSize[2] = mRailOuterSize[2];

  mCradleWall[0] = 0.;  // Inner radius, to be defined from PHOS parameters
  mCradleWall[1] = 65.; // Diff. between outer and inner radii
  mCradleWall[2] = 18.;
  mCradleWall[3] = 270. - 50.;
  mCradleWall[4] = 270. + 50.;

  mCradleWheel[0] = 30.0;
  mCradleWheel[1] = 80.0;
  mCradleWheel[2] = 30.0;
}
