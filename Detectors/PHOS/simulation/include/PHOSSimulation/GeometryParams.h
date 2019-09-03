// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_GEOMETRYPARAMS_H_
#define ALICEO2_PHOS_GEOMETRYPARAMS_H_

#include <string>

#include <RStringView.h>
#include <TNamed.h>
//#include <TVector3.h>

namespace o2
{
namespace phos
{
class GeometryParams : public TNamed
{
 public:
  /// Default constructor
  GeometryParams() = default;

  /// Destructor
  ~GeometryParams() final = default;

  /// Get singleton (create if necessary)
  static GeometryParams* GetInstance(const std::string_view name = "Run2")
  {
    if (!sGeomParam)
      sGeomParam = new GeometryParams(name);
    return sGeomParam;
  }

  // Return general PHOS parameters
  Float_t GetIPtoCrystalSurface() const { return mIPtoCrystalSurface; }
  Float_t GetIPtoOuterCoverDistance() const { return mIPtoOuterCoverDistance; }
  Float_t GetCrystalSize(Int_t index) const { return 2. * mCrystalHalfSize[index]; }
  Int_t GetNPhi() const { return mNPhi; }
  Int_t GetNZ() const { return mNz; }
  Int_t GetNCristalsInModule() const { return mNPhi * mNz; }
  Int_t GetNModules() const { return mNModules; }
  Float_t GetPHOSAngle(Int_t index) const { return mPHOSAngle[index - 1]; }
  Float_t* GetPHOSParams() { return mPHOSParams; }       // Half-sizes of PHOS trapecoid
  Float_t* GetPHOSATBParams() { return mPHOSATBParams; } // Half-sizes of PHOS trapecoid
  Float_t GetOuterBoxSize(Int_t index) const { return 2. * mPHOSParams[index]; }
  Float_t GetCellStep() const { return 2. * mAirCellHalfSize[0]; }

  void GetModuleCenter(Int_t module, Float_t* pos) const
  {
    for (int i = 0; i < 3; i++)
      pos[i] = mModuleCenter[module][i];
  }
  void GetModuleAngle(Int_t module, Float_t angle[3][2]) const
  {
    for (int i = 0; i < 3; i++)
      for (int ian = 0; ian < 2; ian++)
        angle[i][ian] = mModuleAngle[module][i][ian];
  }
  // Return PHOS support geometry parameters
  Float_t GetRailOuterSize(Int_t index) const { return mRailOuterSize[index]; }
  Float_t GetRailPart1(Int_t index) const { return mRailPart1[index]; }
  Float_t GetRailPart2(Int_t index) const { return mRailPart2[index]; }
  Float_t GetRailPart3(Int_t index) const { return mRailPart3[index]; }
  Float_t GetRailPos(Int_t index) const { return mRailPos[index]; }
  Float_t GetRailLength() const { return mRailLength; }
  Float_t GetDistanceBetwRails() const { return mDistanceBetwRails; }
  Float_t GetRailsDistanceFromIP() const { return mRailsDistanceFromIP; }
  Float_t GetRailRoadSize(Int_t index) const { return mRailRoadSize[index]; }
  Float_t GetCradleWallThickness() const { return mCradleWallThickness; }
  Float_t GetCradleWall(Int_t index) const { return mCradleWall[index]; }
  Float_t GetCradleWheel(Int_t index) const { return mCradleWheel[index]; }

  // Return ideal EMC geometry parameters
  const Float_t* GetStripHalfSize() const { return mStripHalfSize; }
  Float_t GetStripWallWidthOut() const { return mStripWallWidthOut; }
  const Float_t* GetAirCellHalfSize() const { return mAirCellHalfSize; }
  const Float_t* GetWrappedHalfSize() const { return mWrappedHalfSize; }
  Float_t GetAirGapLed() const { return mAirGapLed; }
  const Float_t* GetCrystalHalfSize() const { return mCrystalHalfSize; }
  const Float_t* GetSupportPlateHalfSize() const { return mSupportPlateHalfSize; }
  const Float_t* GetSupportPlateInHalfSize() const { return mSupportPlateInHalfSize; }
  Float_t GetSupportPlateThickness() const { return mSupportPlateThickness; }

  const Float_t* GetPreampHalfSize() const { return mPreampHalfSize; }
  const Float_t* GetAPDHalfSize() const { return mPinDiodeHalfSize; }
  const Float_t* GetOuterThermoParams() const { return mOuterThermoParams; }
  const Float_t* GetCoolerHalfSize() const { return mCoolerHalfSize; }
  const Float_t* GetAirGapHalfSize() const { return mAirGapHalfSize; }
  const Float_t* GetInnerThermoHalfSize() const { return mInnerThermoHalfSize; }
  const Float_t* GetAlCoverParams() const { return mAlCoverParams; }
  const Float_t* GetFiberGlassHalfSize() const { return mFiberGlassHalfSize; }
  const Float_t* GetWarmAlCoverHalfSize() const { return mWarmAlCoverHalfSize; }
  const Float_t* GetWarmThermoHalfSize() const { return mWarmThermoHalfSize; }
  const Float_t* GetTSupport1HalfSize() const { return mTSupport1HalfSize; }
  const Float_t* GetTSupport2HalfSize() const { return mTSupport2HalfSize; }
  const Float_t* GetTCables1HalfSize() const { return mTCables1HalfSize; }
  const Float_t* GetTCables2HalfSize() const { return mTCables2HalfSize; }
  Float_t GetTSupportDist() const { return mTSupportDist; }
  const Float_t* GetFrameXHalfSize() const { return mFrameXHalfSize; }
  const Float_t* GetFrameZHalfSize() const { return mFrameZHalfSize; }
  const Float_t* GetFrameXPosition() const { return mFrameXPosition; }
  const Float_t* GetFrameZPosition() const { return mFrameZPosition; }
  const Float_t* GetFGupXHalfSize() const { return mFGupXHalfSize; }
  const Float_t* GetFGupXPosition() const { return mFGupXPosition; }
  const Float_t* GetFGupZHalfSize() const { return mFGupZHalfSize; }
  const Float_t* GetFGupZPosition() const { return mFGupZPosition; }
  const Float_t* GetFGlowXHalfSize() const { return mFGlowXHalfSize; }
  const Float_t* GetFGlowXPosition() const { return mFGlowXPosition; }
  const Float_t* GetFGlowZHalfSize() const { return mFGlowZHalfSize; }
  const Float_t* GetFGlowZPosition() const { return mFGlowZPosition; }
  const Float_t* GetFEEAirHalfSize() const { return mFEEAirHalfSize; }
  const Float_t* GetFEEAirPosition() const { return mFEEAirPosition; }
  const Float_t* GetEMCParams() const { return mEMCParams; }
  const Float_t GetDistATBtoModule() const { return mzAirTightBoxToTopModuleDist; }
  const Float_t GetATBWallWidth() const { return mATBoxWall; }

  Int_t GetNCellsXInStrip() const { return mNCellsXInStrip; }
  Int_t GetNCellsZInStrip() const { return mNCellsZInStrip; }
  Int_t GetNStripX() const { return mNStripX; }
  Int_t GetNStripZ() const { return mNStripZ; }
  Int_t GetNTSuppots() const { return mNTSupports; }

 private:
  ///
  /// Main constructor
  ///
  /// Geometry configuration: Run2,...
  GeometryParams(const std::string_view name);

  static GeometryParams* sGeomParam; ///< Pointer to the unique instance of the singleton

  // General PHOS modules parameters
  Int_t mNModules;               ///< Number of PHOS modules
  Float_t mAngle;                ///< Position angles between modules
  Float_t mPHOSAngle[4];         ///< Position angles of modules
  Float_t mPHOSParams[4];        ///< Half-sizes of PHOS trapecoid
  Float_t mPHOSATBParams[4];     ///< Half-sizes of (air-filled) inner part of PHOS air tight box
  Float_t mCrystalShift;         ///< Distance from crystal center to front surface
  Float_t mCryCellShift;         ///< Distance from crystal center to front surface
  Float_t mModuleCenter[5][3];   ///< xyz-position of the module center
  Float_t mModuleAngle[5][3][2]; ///< polar and azymuth angles for 3 axes of modules

  // EMC geometry parameters

  Float_t mStripHalfSize[3];          ///< Strip unit size/2
  Float_t mAirCellHalfSize[3];        ///< geometry parameter
  Float_t mWrappedHalfSize[3];        ///< geometry parameter
  Float_t mSupportPlateHalfSize[3];   ///< geometry parameter
  Float_t mSupportPlateInHalfSize[3]; ///< geometry parameter
  Float_t mCrystalHalfSize[3];        ///< crystal size/2
  Float_t mAirGapLed;                 ///< geometry parameter
  Float_t mStripWallWidthOut;         ///< Side to another strip
  Float_t mStripWallWidthIn;          ///< geometry parameter
  Float_t mTyvecThickness;            ///< geometry parameter
  Float_t mTSupport1HalfSize[3];      ///< geometry parameter
  Float_t mTSupport2HalfSize[3];      ///< geometry parameter
  Float_t mPreampHalfSize[3];         ///< geometry parameter
  Float_t mPinDiodeHalfSize[3];       ///< Size of the PIN Diode

  Float_t mOuterThermoParams[4];   // geometry parameter
  Float_t mCoolerHalfSize[3];      // geometry parameter
  Float_t mAirGapHalfSize[3];      // geometry parameter
  Float_t mInnerThermoHalfSize[3]; // geometry parameter
  Float_t mAlCoverParams[4];       // geometry parameter
  Float_t mFiberGlassHalfSize[3];  // geometry parameter

  Float_t mInnerThermoWidthX;      // geometry parameter
  Float_t mInnerThermoWidthY;      // geometry parameter
  Float_t mInnerThermoWidthZ;      // geometry parameter
  Float_t mAirGapWidthX;           // geometry parameter
  Float_t mAirGapWidthY;           // geometry parameter
  Float_t mAirGapWidthZ;           // geometry parameter
  Float_t mCoolerWidthX;           // geometry parameter
  Float_t mCoolerWidthY;           // geometry parameter
  Float_t mCoolerWidthZ;           // geometry parameter
  Float_t mAlCoverThickness;       // geometry parameter
  Float_t mOuterThermoWidthXUp;    // geometry parameter
  Float_t mOuterThermoWidthXLow;   // geometry parameter
  Float_t mOuterThermoWidthY;      // geometry parameter
  Float_t mOuterThermoWidthZ;      // geometry parameter
  Float_t mAlFrontCoverX;          // geometry parameter
  Float_t mAlFrontCoverZ;          // geometry parameter
  Float_t mFiberGlassSup2X;        // geometry parameter
  Float_t mFiberGlassSup1X;        // geometry parameter
  Float_t mFrameHeight;            // geometry parameter
  Float_t mFrameThickness;         // geometry parameter
  Float_t mAirSpaceFeeX;           // geometry parameter
  Float_t mAirSpaceFeeZ;           // geometry parameter
  Float_t mAirSpaceFeeY;           // geometry parameter
  Float_t mTCables2HalfSize[3];    // geometry parameter
  Float_t mTCables1HalfSize[3];    // geometry parameter
  Float_t mWarmUpperThickness;     // geometry parameter
  Float_t mWarmBottomThickness;    // geometry parameter
  Float_t mWarmAlCoverWidthX;      // geometry parameter
  Float_t mWarmAlCoverWidthY;      // geometry parameter
  Float_t mWarmAlCoverWidthZ;      // geometry parameter
  Float_t mWarmAlCoverHalfSize[3]; // geometry parameter
  Float_t mWarmThermoHalfSize[3];  // geometry parameter
  Float_t mFiberGlassSup1Y;        // geometry parameter
  Float_t mFiberGlassSup2Y;        // geometry parameter
  Float_t mTSupportDist;           // geometry parameter
  Float_t mTSupport1Thickness;     // geometry parameter
  Float_t mTSupport2Thickness;     // geometry parameter
  Float_t mTSupport1Width;         // geometry parameter
  Float_t mTSupport2Width;         // geometry parameter
  Float_t mFrameXHalfSize[3];      // geometry parameter
  Float_t mFrameZHalfSize[3];      // geometry parameter
  Float_t mFrameXPosition[3];      // geometry parameter
  Float_t mFrameZPosition[3];      // geometry parameter
  Float_t mFGupXHalfSize[3];       // geometry parameter
  Float_t mFGupXPosition[3];       // geometry parameter
  Float_t mFGupZHalfSize[3];       // geometry parameter
  Float_t mFGupZPosition[3];       // geometry parameter
  Float_t mFGlowXHalfSize[3];      // geometry parameter
  Float_t mFGlowXPosition[3];      // geometry parameter
  Float_t mFGlowZHalfSize[3];      // geometry parameter
  Float_t mFGlowZPosition[3];      // geometry parameter
  Float_t mFEEAirHalfSize[3];      // geometry parameter
  Float_t mFEEAirPosition[3];      // geometry parameter
  Float_t mEMCParams[4];           // geometry parameter
  Float_t mIPtoOuterCoverDistance; ///< Distances from interaction point to outer cover
  Float_t mIPtoCrystalSurface;     ///< Distances from interaction point to Xtal surface

  Float_t mSupportPlateThickness;       ///< Thickness of the Aluminium support plate for Strip
  Float_t mzAirTightBoxToTopModuleDist; ///< Distance between PHOS upper surface and inner part of Air Tight Box
  Float_t mATBoxWall;                   ///< width of the wall of air tight box

  Int_t mNCellsXInStrip; ///< Number of cells in a strip unit in X
  Int_t mNCellsZInStrip; ///< Number of cells in a strip unit in Z
  Int_t mNStripX;        ///< Number of strip units in X
  Int_t mNStripZ;        ///< Number of strip units in Z
  Int_t mNTSupports;     ///< geometry parameter
  Int_t mNPhi;           ///< Number of crystal units in X (phi) direction
  Int_t mNz;             ///< Number of crystal units in Z direction

  // Support geometry parameters
  Float_t mRailOuterSize[3];    ///< Outer size of a rail                 +-------+
  Float_t mRailPart1[3];        ///< Upper & bottom parts of the rail     |--+ +--|
  Float_t mRailPart2[3];        ///< Vertical middle parts of the rail       | |
  Float_t mRailPart3[3];        ///< Vertical upper parts of the rail        | |
  Float_t mRailPos[3];          ///< Rail position vs. the ALICE center   |--+ +--|
  Float_t mRailLength;          ///< Length of the rail under the support +-------+
  Float_t mDistanceBetwRails;   ///< Distance between rails
  Float_t mRailsDistanceFromIP; ///< Distance of rails from IP
  Float_t mRailRoadSize[3];     ///< Outer size of the dummy box with rails
  Float_t mCradleWallThickness; ///< PHOS cradle wall thickness
  Float_t mCradleWall[5];       ///< Size of the wall of the PHOS cradle (shape TUBS)
  Float_t mCradleWheel[3];      ///< "Wheels" by which the cradle rolls over the rails

  ClassDefOverride(GeometryParams, 1);
};
} // namespace phos
} // namespace o2
#endif
