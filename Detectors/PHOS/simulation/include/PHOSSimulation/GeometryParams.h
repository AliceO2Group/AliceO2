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
class GeometryParams final : public TNamed
{
 public:
  /// Default constructor
  GeometryParams() = default;

  /// Destructor
  ~GeometryParams() final = default;

  /// get singleton (create if necessary)
  static GeometryParams* GetInstance(const std::string_view name = "Run2")
  {
    if (!sGeomParam) {
      sGeomParam = new GeometryParams(name);
    }
    return sGeomParam;
  }

  // Return general PHOS parameters
  double getIPtoCrystalSurface() const { return mIPtoCrystalSurface; }
  double getIPtoOuterCoverDistance() const { return mIPtoOuterCoverDistance; }
  double getCrystalSize(int index) const { return 2. * mCrystalHalfSize[index]; }
  int getNPhi() const { return mNPhi; }
  int getNZ() const { return mNz; }
  int getNCristalsInModule() const { return mNPhi * mNz; }
  int getNModules() const { return mNModules; }
  double getPHOSAngle(int index) const { return mPHOSAngle[index]; }
  double* getPHOSParams() { return mPHOSParams; }       // Half-sizes of PHOS trapecoid
  double* getPHOSATBParams() { return mPHOSATBParams; } // Half-sizes of PHOS trapecoid
  double getOuterBoxSize(int index) const { return 2. * mPHOSParams[index]; }
  double getCellStep() const { return 2. * mAirCellHalfSize[0]; }

  void getModuleCenter(int module, double* pos) const
  {
    for (int i = 0; i < 3; i++) {
      pos[i] = mModuleCenter[module][i];
    }
  }
  void getModuleAngle(int module, double angle[3][2]) const
  {
    for (int i = 0; i < 3; i++) {
      for (int ian = 0; ian < 2; ian++) {
        angle[i][ian] = mModuleAngle[module][i][ian];
      }
    }
  }
  // Return PHOS support geometry parameters
  double getRailOuterSize(int index) const { return mRailOuterSize[index]; }
  double getRailPart1(int index) const { return mRailPart1[index]; }
  double getRailPart2(int index) const { return mRailPart2[index]; }
  double getRailPart3(int index) const { return mRailPart3[index]; }
  double getRailPos(int index) const { return mRailPos[index]; }
  double getRailLength() const { return mRailLength; }
  double getDistanceBetwRails() const { return mDistanceBetwRails; }
  double getRailsDistanceFromIP() const { return mRailsDistanceFromIP; }
  double getRailRoadSize(int index) const { return mRailRoadSize[index]; }
  double getModuleCraddleGap() const { return mModuleCraddleGap; }
  double getCradleWallThickness() const { return mCradleWallThickness; }
  double getCradleWall(int index) const { return mCradleWall[index]; }
  double getCradleWheel(int index) const { return mCradleWheel[index]; }

  // Return ideal EMC geometry parameters
  const double* getStripHalfSize() const { return mStripHalfSize; }
  double getStripWallWidthOut() const { return mStripWallWidthOut; }
  const double* getAirCellHalfSize() const { return mAirCellHalfSize; }
  const double* getWrappedHalfSize() const { return mWrappedHalfSize; }
  double getAirGapLed() const { return mAirGapLed; }
  const double* getCrystalHalfSize() const { return mCrystalHalfSize; }
  const double* getSupportPlateHalfSize() const { return mSupportPlateHalfSize; }
  const double* getSupportPlateInHalfSize() const { return mSupportPlateInHalfSize; }
  double getSupportPlateThickness() const { return mSupportPlateThickness; }

  const double* getPreampHalfSize() const { return mPreampHalfSize; }
  const double* getAPDHalfSize() const { return mPinDiodeHalfSize; }
  const double* getOuterThermoParams() const { return mOuterThermoParams; }
  const double* getCoolerHalfSize() const { return mCoolerHalfSize; }
  const double* getAirGapHalfSize() const { return mAirGapHalfSize; }
  const double* getInnerThermoHalfSize() const { return mInnerThermoHalfSize; }
  const double* getAlCoverParams() const { return mAlCoverParams; }
  const double* getFiberGlassHalfSize() const { return mFiberGlassHalfSize; }
  const double* getWarmAlCoverHalfSize() const { return mWarmAlCoverHalfSize; }
  const double* getWarmThermoHalfSize() const { return mWarmThermoHalfSize; }
  const double* getTSupport1HalfSize() const { return mTSupport1HalfSize; }
  const double* getTSupport2HalfSize() const { return mTSupport2HalfSize; }
  const double* getTCables1HalfSize() const { return mTCables1HalfSize; }
  const double* getTCables2HalfSize() const { return mTCables2HalfSize; }
  double getTSupportDist() const { return mTSupportDist; }
  const double* getFrameXHalfSize() const { return mFrameXHalfSize; }
  const double* getFrameZHalfSize() const { return mFrameZHalfSize; }
  const double* getFrameXPosition() const { return mFrameXPosition; }
  const double* getFrameZPosition() const { return mFrameZPosition; }
  const double* getFGupXHalfSize() const { return mFGupXHalfSize; }
  const double* getFGupXPosition() const { return mFGupXPosition; }
  const double* getFGupZHalfSize() const { return mFGupZHalfSize; }
  const double* getFGupZPosition() const { return mFGupZPosition; }
  const double* getFGlowXHalfSize() const { return mFGlowXHalfSize; }
  const double* getFGlowXPosition() const { return mFGlowXPosition; }
  const double* getFGlowZHalfSize() const { return mFGlowZHalfSize; }
  const double* getFGlowZPosition() const { return mFGlowZPosition; }
  const double* getFEEAirHalfSize() const { return mFEEAirHalfSize; }
  const double* getFEEAirPosition() const { return mFEEAirPosition; }
  const double* getEMCParams() const { return mEMCParams; }
  double getDistATBtoModule() const { return mzAirTightBoxToTopModuleDist; }
  double getATBWallWidth() const { return mATBoxWall; }

  int getNCellsXInStrip() const { return mNCellsXInStrip; }
  int getNCellsZInStrip() const { return mNCellsZInStrip; }
  int getNStripX() const { return mNStripX; }
  int getNStripZ() const { return mNStripZ; }
  int getNTSuppots() const { return mNTSupports; }

 private:
  ///
  /// Main constructor
  ///
  /// Geometry configuration: Run2,...
  GeometryParams(const std::string_view name);

  static GeometryParams* sGeomParam; ///< Pointer to the unique instance of the singleton

  // General PHOS modules parameters
  int mNModules;               ///< Number of PHOS modules
  double mAngle;               ///< Position angles between modules
  double mPHOSAngle[5];        ///< Position angles of modules
  double mPHOSParams[4];       ///< Half-sizes of PHOS trapecoid
  double mPHOSATBParams[4];    ///< Half-sizes of (air-filled) inner part of PHOS air tight box
  double mCrystalShift;        ///< Distance from crystal center to front surface
  double mCryCellShift;        ///< Distance from crystal center to front surface
  double mModuleCenter[5][3];  ///< xyz-position of the module center
  double mModuleAngle[5][3][2]; ///< polar and azymuth angles for 3 axes of modules

  // EMC geometry parameters

  double mStripHalfSize[3];          ///< Strip unit size/2
  double mAirCellHalfSize[3];        ///< geometry parameter
  double mWrappedHalfSize[3];        ///< geometry parameter
  double mSupportPlateHalfSize[3];   ///< geometry parameter
  double mSupportPlateInHalfSize[3]; ///< geometry parameter
  double mCrystalHalfSize[3];        ///< crystal size/2
  double mAirGapLed;                 ///< geometry parameter
  double mStripWallWidthOut;         ///< Side to another strip
  double mStripWallWidthIn;          ///< geometry parameter
  double mTyvecThickness;            ///< geometry parameter
  double mTSupport1HalfSize[3];      ///< geometry parameter
  double mTSupport2HalfSize[3];      ///< geometry parameter
  double mPreampHalfSize[3];         ///< geometry parameter
  double mPinDiodeHalfSize[3];       ///< Size of the PIN Diode

  double mOuterThermoParams[4];   // geometry parameter
  double mCoolerHalfSize[3];      // geometry parameter
  double mAirGapHalfSize[3];      // geometry parameter
  double mInnerThermoHalfSize[3]; // geometry parameter
  double mAlCoverParams[4];       // geometry parameter
  double mFiberGlassHalfSize[3];  // geometry parameter

  double mInnerThermoWidthX;      // geometry parameter
  double mInnerThermoWidthY;      // geometry parameter
  double mInnerThermoWidthZ;      // geometry parameter
  double mAirGapWidthX;           // geometry parameter
  double mAirGapWidthY;           // geometry parameter
  double mAirGapWidthZ;           // geometry parameter
  double mCoolerWidthX;           // geometry parameter
  double mCoolerWidthY;           // geometry parameter
  double mCoolerWidthZ;           // geometry parameter
  double mAlCoverThickness;       // geometry parameter
  double mOuterThermoWidthXUp;    // geometry parameter
  double mOuterThermoWidthXLow;   // geometry parameter
  double mOuterThermoWidthY;      // geometry parameter
  double mOuterThermoWidthZ;      // geometry parameter
  double mAlFrontCoverX;          // geometry parameter
  double mAlFrontCoverZ;          // geometry parameter
  double mFiberGlassSup2X;        // geometry parameter
  double mFiberGlassSup1X;        // geometry parameter
  double mFrameHeight;            // geometry parameter
  double mFrameThickness;         // geometry parameter
  double mAirSpaceFeeX;           // geometry parameter
  double mAirSpaceFeeZ;           // geometry parameter
  double mAirSpaceFeeY;           // geometry parameter
  double mTCables2HalfSize[3];    // geometry parameter
  double mTCables1HalfSize[3];    // geometry parameter
  double mWarmUpperThickness;     // geometry parameter
  double mWarmBottomThickness;    // geometry parameter
  double mWarmAlCoverWidthX;      // geometry parameter
  double mWarmAlCoverWidthY;      // geometry parameter
  double mWarmAlCoverWidthZ;      // geometry parameter
  double mWarmAlCoverHalfSize[3]; // geometry parameter
  double mWarmThermoHalfSize[3];  // geometry parameter
  double mFiberGlassSup1Y;        // geometry parameter
  double mFiberGlassSup2Y;        // geometry parameter
  double mTSupportDist;           // geometry parameter
  double mTSupport1Thickness;     // geometry parameter
  double mTSupport2Thickness;     // geometry parameter
  double mTSupport1Width;         // geometry parameter
  double mTSupport2Width;         // geometry parameter
  double mFrameXHalfSize[3];      // geometry parameter
  double mFrameZHalfSize[3];      // geometry parameter
  double mFrameXPosition[3];      // geometry parameter
  double mFrameZPosition[3];      // geometry parameter
  double mFGupXHalfSize[3];       // geometry parameter
  double mFGupXPosition[3];       // geometry parameter
  double mFGupZHalfSize[3];       // geometry parameter
  double mFGupZPosition[3];       // geometry parameter
  double mFGlowXHalfSize[3];      // geometry parameter
  double mFGlowXPosition[3];      // geometry parameter
  double mFGlowZHalfSize[3];      // geometry parameter
  double mFGlowZPosition[3];      // geometry parameter
  double mFEEAirHalfSize[3];      // geometry parameter
  double mFEEAirPosition[3];      // geometry parameter
  double mEMCParams[4];           // geometry parameter
  double mIPtoOuterCoverDistance; ///< Distances from interaction point to outer cover
  double mIPtoCrystalSurface;     ///< Distances from interaction point to Xtal surface

  double mSupportPlateThickness;       ///< Thickness of the Aluminium support plate for Strip
  double mzAirTightBoxToTopModuleDist; ///< Distance between PHOS upper surface and inner part of Air Tight Box
  double mATBoxWall;                   ///< width of the wall of air tight box

  int mNCellsXInStrip; ///< Number of cells in a strip unit in X
  int mNCellsZInStrip; ///< Number of cells in a strip unit in Z
  int mNStripX;        ///< Number of strip units in X
  int mNStripZ;        ///< Number of strip units in Z
  int mNTSupports;     ///< geometry parameter
  int mNPhi;           ///< Number of crystal units in X (phi) direction
  int mNz;             ///< Number of crystal units in Z direction

  // Support geometry parameters
  double mRailOuterSize[3];    ///< Outer size of a rail                 +-------+
  double mRailPart1[3];        ///< Upper & bottom parts of the rail     |--+ +--|
  double mRailPart2[3];        ///< Vertical middle parts of the rail       | |
  double mRailPart3[3];        ///< Vertical upper parts of the rail        | |
  double mRailPos[3];          ///< Rail position vs. the ALICE center   |--+ +--|
  double mRailLength;          ///< Length of the rail under the support +-------+
  double mDistanceBetwRails;   ///< Distance between rails
  double mRailsDistanceFromIP; ///< Distance of rails from IP
  double mRailRoadSize[3];     ///< Outer size of the dummy box with rails
  double mCradleWallThickness; ///< PHOS cradle wall thickness
  double mModuleCraddleGap;    ///< gap between PHOS module and craddle inner wall
  double mCradleWall[5];       ///< Size of the wall of the PHOS cradle (shape TUBS)
  double mCradleWheel[3];      ///< "Wheels" by which the cradle rolls over the rails

  ClassDefOverride(GeometryParams, 2);
};
} // namespace phos
} // namespace o2
#endif
