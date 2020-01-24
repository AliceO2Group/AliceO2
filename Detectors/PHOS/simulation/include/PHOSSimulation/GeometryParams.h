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

  /// get singleton (create if necessary)
  static GeometryParams* GetInstance(const std::string_view name = "Run2")
  {
    if (!sGeomParam)
      sGeomParam = new GeometryParams(name);
    return sGeomParam;
  }

  // Return general PHOS parameters
  float getIPtoCrystalSurface() const { return mIPtoCrystalSurface; }
  float getIPtoOuterCoverDistance() const { return mIPtoOuterCoverDistance; }
  float getCrystalSize(int index) const { return 2. * mCrystalHalfSize[index]; }
  int getNPhi() const { return mNPhi; }
  int getNZ() const { return mNz; }
  int getNCristalsInModule() const { return mNPhi * mNz; }
  int getNModules() const { return mNModules; }
  float getPHOSAngle(int index) const { return mPHOSAngle[index - 1]; }
  float* getPHOSParams() { return mPHOSParams; }       // Half-sizes of PHOS trapecoid
  float* getPHOSATBParams() { return mPHOSATBParams; } // Half-sizes of PHOS trapecoid
  float getOuterBoxSize(int index) const { return 2. * mPHOSParams[index]; }
  float getCellStep() const { return 2. * mAirCellHalfSize[0]; }

  void getModuleCenter(int module, float* pos) const
  {
    for (int i = 0; i < 3; i++)
      pos[i] = mModuleCenter[module][i];
  }
  void getModuleAngle(int module, float angle[3][2]) const
  {
    for (int i = 0; i < 3; i++)
      for (int ian = 0; ian < 2; ian++)
        angle[i][ian] = mModuleAngle[module][i][ian];
  }
  // Return PHOS support geometry parameters
  float getRailOuterSize(int index) const { return mRailOuterSize[index]; }
  float getRailPart1(int index) const { return mRailPart1[index]; }
  float getRailPart2(int index) const { return mRailPart2[index]; }
  float getRailPart3(int index) const { return mRailPart3[index]; }
  float getRailPos(int index) const { return mRailPos[index]; }
  float getRailLength() const { return mRailLength; }
  float getDistanceBetwRails() const { return mDistanceBetwRails; }
  float getRailsDistanceFromIP() const { return mRailsDistanceFromIP; }
  float getRailRoadSize(int index) const { return mRailRoadSize[index]; }
  float getCradleWallThickness() const { return mCradleWallThickness; }
  float getCradleWall(int index) const { return mCradleWall[index]; }
  float getCradleWheel(int index) const { return mCradleWheel[index]; }

  // Return ideal EMC geometry parameters
  const float* getStripHalfSize() const { return mStripHalfSize; }
  float getStripWallWidthOut() const { return mStripWallWidthOut; }
  const float* getAirCellHalfSize() const { return mAirCellHalfSize; }
  const float* getWrappedHalfSize() const { return mWrappedHalfSize; }
  float getAirGapLed() const { return mAirGapLed; }
  const float* getCrystalHalfSize() const { return mCrystalHalfSize; }
  const float* getSupportPlateHalfSize() const { return mSupportPlateHalfSize; }
  const float* getSupportPlateInHalfSize() const { return mSupportPlateInHalfSize; }
  float getSupportPlateThickness() const { return mSupportPlateThickness; }

  const float* getPreampHalfSize() const { return mPreampHalfSize; }
  const float* getAPDHalfSize() const { return mPinDiodeHalfSize; }
  const float* getOuterThermoParams() const { return mOuterThermoParams; }
  const float* getCoolerHalfSize() const { return mCoolerHalfSize; }
  const float* getAirGapHalfSize() const { return mAirGapHalfSize; }
  const float* getInnerThermoHalfSize() const { return mInnerThermoHalfSize; }
  const float* getAlCoverParams() const { return mAlCoverParams; }
  const float* getFiberGlassHalfSize() const { return mFiberGlassHalfSize; }
  const float* getWarmAlCoverHalfSize() const { return mWarmAlCoverHalfSize; }
  const float* getWarmThermoHalfSize() const { return mWarmThermoHalfSize; }
  const float* getTSupport1HalfSize() const { return mTSupport1HalfSize; }
  const float* getTSupport2HalfSize() const { return mTSupport2HalfSize; }
  const float* getTCables1HalfSize() const { return mTCables1HalfSize; }
  const float* getTCables2HalfSize() const { return mTCables2HalfSize; }
  float getTSupportDist() const { return mTSupportDist; }
  const float* getFrameXHalfSize() const { return mFrameXHalfSize; }
  const float* getFrameZHalfSize() const { return mFrameZHalfSize; }
  const float* getFrameXPosition() const { return mFrameXPosition; }
  const float* getFrameZPosition() const { return mFrameZPosition; }
  const float* getFGupXHalfSize() const { return mFGupXHalfSize; }
  const float* getFGupXPosition() const { return mFGupXPosition; }
  const float* getFGupZHalfSize() const { return mFGupZHalfSize; }
  const float* getFGupZPosition() const { return mFGupZPosition; }
  const float* getFGlowXHalfSize() const { return mFGlowXHalfSize; }
  const float* getFGlowXPosition() const { return mFGlowXPosition; }
  const float* getFGlowZHalfSize() const { return mFGlowZHalfSize; }
  const float* getFGlowZPosition() const { return mFGlowZPosition; }
  const float* getFEEAirHalfSize() const { return mFEEAirHalfSize; }
  const float* getFEEAirPosition() const { return mFEEAirPosition; }
  const float* getEMCParams() const { return mEMCParams; }
  float getDistATBtoModule() const { return mzAirTightBoxToTopModuleDist; }
  float getATBWallWidth() const { return mATBoxWall; }

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
  float mAngle;                ///< Position angles between modules
  float mPHOSAngle[4];         ///< Position angles of modules
  float mPHOSParams[4];        ///< Half-sizes of PHOS trapecoid
  float mPHOSATBParams[4];     ///< Half-sizes of (air-filled) inner part of PHOS air tight box
  float mCrystalShift;         ///< Distance from crystal center to front surface
  float mCryCellShift;         ///< Distance from crystal center to front surface
  float mModuleCenter[5][3];   ///< xyz-position of the module center
  float mModuleAngle[5][3][2]; ///< polar and azymuth angles for 3 axes of modules

  // EMC geometry parameters

  float mStripHalfSize[3];          ///< Strip unit size/2
  float mAirCellHalfSize[3];        ///< geometry parameter
  float mWrappedHalfSize[3];        ///< geometry parameter
  float mSupportPlateHalfSize[3];   ///< geometry parameter
  float mSupportPlateInHalfSize[3]; ///< geometry parameter
  float mCrystalHalfSize[3];        ///< crystal size/2
  float mAirGapLed;                 ///< geometry parameter
  float mStripWallWidthOut;         ///< Side to another strip
  float mStripWallWidthIn;          ///< geometry parameter
  float mTyvecThickness;            ///< geometry parameter
  float mTSupport1HalfSize[3];      ///< geometry parameter
  float mTSupport2HalfSize[3];      ///< geometry parameter
  float mPreampHalfSize[3];         ///< geometry parameter
  float mPinDiodeHalfSize[3];       ///< Size of the PIN Diode

  float mOuterThermoParams[4];   // geometry parameter
  float mCoolerHalfSize[3];      // geometry parameter
  float mAirGapHalfSize[3];      // geometry parameter
  float mInnerThermoHalfSize[3]; // geometry parameter
  float mAlCoverParams[4];       // geometry parameter
  float mFiberGlassHalfSize[3];  // geometry parameter

  float mInnerThermoWidthX;      // geometry parameter
  float mInnerThermoWidthY;      // geometry parameter
  float mInnerThermoWidthZ;      // geometry parameter
  float mAirGapWidthX;           // geometry parameter
  float mAirGapWidthY;           // geometry parameter
  float mAirGapWidthZ;           // geometry parameter
  float mCoolerWidthX;           // geometry parameter
  float mCoolerWidthY;           // geometry parameter
  float mCoolerWidthZ;           // geometry parameter
  float mAlCoverThickness;       // geometry parameter
  float mOuterThermoWidthXUp;    // geometry parameter
  float mOuterThermoWidthXLow;   // geometry parameter
  float mOuterThermoWidthY;      // geometry parameter
  float mOuterThermoWidthZ;      // geometry parameter
  float mAlFrontCoverX;          // geometry parameter
  float mAlFrontCoverZ;          // geometry parameter
  float mFiberGlassSup2X;        // geometry parameter
  float mFiberGlassSup1X;        // geometry parameter
  float mFrameHeight;            // geometry parameter
  float mFrameThickness;         // geometry parameter
  float mAirSpaceFeeX;           // geometry parameter
  float mAirSpaceFeeZ;           // geometry parameter
  float mAirSpaceFeeY;           // geometry parameter
  float mTCables2HalfSize[3];    // geometry parameter
  float mTCables1HalfSize[3];    // geometry parameter
  float mWarmUpperThickness;     // geometry parameter
  float mWarmBottomThickness;    // geometry parameter
  float mWarmAlCoverWidthX;      // geometry parameter
  float mWarmAlCoverWidthY;      // geometry parameter
  float mWarmAlCoverWidthZ;      // geometry parameter
  float mWarmAlCoverHalfSize[3]; // geometry parameter
  float mWarmThermoHalfSize[3];  // geometry parameter
  float mFiberGlassSup1Y;        // geometry parameter
  float mFiberGlassSup2Y;        // geometry parameter
  float mTSupportDist;           // geometry parameter
  float mTSupport1Thickness;     // geometry parameter
  float mTSupport2Thickness;     // geometry parameter
  float mTSupport1Width;         // geometry parameter
  float mTSupport2Width;         // geometry parameter
  float mFrameXHalfSize[3];      // geometry parameter
  float mFrameZHalfSize[3];      // geometry parameter
  float mFrameXPosition[3];      // geometry parameter
  float mFrameZPosition[3];      // geometry parameter
  float mFGupXHalfSize[3];       // geometry parameter
  float mFGupXPosition[3];       // geometry parameter
  float mFGupZHalfSize[3];       // geometry parameter
  float mFGupZPosition[3];       // geometry parameter
  float mFGlowXHalfSize[3];      // geometry parameter
  float mFGlowXPosition[3];      // geometry parameter
  float mFGlowZHalfSize[3];      // geometry parameter
  float mFGlowZPosition[3];      // geometry parameter
  float mFEEAirHalfSize[3];      // geometry parameter
  float mFEEAirPosition[3];      // geometry parameter
  float mEMCParams[4];           // geometry parameter
  float mIPtoOuterCoverDistance; ///< Distances from interaction point to outer cover
  float mIPtoCrystalSurface;     ///< Distances from interaction point to Xtal surface

  float mSupportPlateThickness;       ///< Thickness of the Aluminium support plate for Strip
  float mzAirTightBoxToTopModuleDist; ///< Distance between PHOS upper surface and inner part of Air Tight Box
  float mATBoxWall;                   ///< width of the wall of air tight box

  int mNCellsXInStrip; ///< Number of cells in a strip unit in X
  int mNCellsZInStrip; ///< Number of cells in a strip unit in Z
  int mNStripX;        ///< Number of strip units in X
  int mNStripZ;        ///< Number of strip units in Z
  int mNTSupports;     ///< geometry parameter
  int mNPhi;           ///< Number of crystal units in X (phi) direction
  int mNz;             ///< Number of crystal units in Z direction

  // Support geometry parameters
  float mRailOuterSize[3];    ///< Outer size of a rail                 +-------+
  float mRailPart1[3];        ///< Upper & bottom parts of the rail     |--+ +--|
  float mRailPart2[3];        ///< Vertical middle parts of the rail       | |
  float mRailPart3[3];        ///< Vertical upper parts of the rail        | |
  float mRailPos[3];          ///< Rail position vs. the ALICE center   |--+ +--|
  float mRailLength;          ///< Length of the rail under the support +-------+
  float mDistanceBetwRails;   ///< Distance between rails
  float mRailsDistanceFromIP; ///< Distance of rails from IP
  float mRailRoadSize[3];     ///< Outer size of the dummy box with rails
  float mCradleWallThickness; ///< PHOS cradle wall thickness
  float mCradleWall[5];       ///< Size of the wall of the PHOS cradle (shape TUBS)
  float mCradleWheel[3];      ///< "Wheels" by which the cradle rolls over the rails

  ClassDefOverride(GeometryParams, 1);
};
} // namespace phos
} // namespace o2
#endif
