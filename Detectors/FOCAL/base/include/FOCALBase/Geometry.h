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

#ifndef ALICEO2_FOCAL_GEOMETRY_H_
#define ALICEO2_FOCAL_GEOMETRY_H_

#include <string>
#include <vector>
#include <array>
#include <list>
#include <tuple>

#include "FOCALBase/Composition.h"

namespace o2
{
namespace focal
{

class VirtualSegment
{

 public:
  VirtualSegment()
  {
  }

  VirtualSegment& operator=(const VirtualSegment& segment)
  {
    if (this != &segment) {
      mMinLayer = segment.mMinLayer;
      mMaxLayer = segment.mMaxLayer;
      mPadSize = segment.mPadSize;
      mRelativeSensitiveThickness = segment.mRelativeSensitiveThickness;
      mIsPixel = segment.mIsPixel;
      mPixelTreshold = segment.mPixelTreshold;
    }
    return *this;
  }

  ~VirtualSegment() {}

  int mMinLayer = -1;
  int mMaxLayer = -1;
  float mPadSize = -1.0;
  float mRelativeSensitiveThickness = -1.0;
  int mIsPixel = 0; // 0: pad or strip; 1: pixel; 2: HCAL
  float mPixelTreshold = -1.0;
}; // class VirtualSegment

class Geometry
{
 public:
  Geometry() = default;
  Geometry(Geometry* geo);
  Geometry(const Geometry& geo) = default;
  Geometry& operator=(const Geometry& geo) = default;
  ~Geometry() = default;

  static Geometry* getInstance();
  static Geometry* getInstance(const std::string name);

  void init(const std::string geoFile);
  void init();
  void buildComposition();
  void setParameters(const std::string geoFile);
  void setParameters();

  // geometry helper functions
  // TODO: return tuples instead of using references
  std::tuple<double /*x*/, double /*y*/, double /*z*/> getGeoPadCenter(int tower, int layer, int stack, int row, int col) const;
  std::tuple<double /*x*/, double /*y*/> getGeoPadCenterLocal(int towerX, int towerY, int row, int col) const;
  std::tuple<double /*x*/, double /*y*/> getGeoPixCenterLocal(int row, int col) const;
  std::tuple<double /*x*/, double /*y*/, double /*z*/> getGeoPixelCenter(int pixel_id, int tower, int layer, int stack, int row, int col) const;
  std::tuple<double /*x*/, double /*y*/, double /*z*/> getGeoCompositionCenter(int tower, int layer, int stack) const;
  std::tuple<int /*row*/, int /*col*/, int /*stack*/, int /*layer*/, int /*seg*/, int /*waferx*/, int /*wafery*/> getPadPositionId2RowColStackLayer(int id) const;
  int getPixelNumber(int vol0, int vol1, int vol2, double x, double y) const;
  std::tuple<double /*x*/, double /*y*/, double /*z*/> getGeoTowerCenter(int tower, int segment = -1) const;
  bool disabledTower(int tower);
  std::tuple<bool /*statusCode*/, int /*col*/, int /*row*/, int /*layer*/, int /*segment*/> getVirtualInfo(double x, double y, double z) const;
  std::tuple<bool /*statusCode*/, double /*x*/, double /*y*/, double /*z*/> getXYZFromColRowSeg(int col, int row, int segment) const;
  std::tuple<bool /*statusCode*/, int /*nCol*/, int /*nRow*/> getVirtualNColRow(int segment) const;
  std::tuple<bool /*statusCode*/, int /*layer*/, int /*segment*/> getVirtualLayerSegment(float z) const;
  std::tuple<bool /*statusCode*/, int /*segment*/> getVirtualSegmentFromLayer(int layer) const;
  int getVirtualSegment(float z) const;

  // getters
  int getNumberOfPads() const { return mGlobal_PAD_NX * mGlobal_PAD_NY; }
  int getNumberOfPADsInX() const { return mGlobal_PAD_NX_Tower; }
  int getNumberOfPADsInY() const { return mGlobal_PAD_NY_Tower; }
  int getNumberOfPIXsInX() const { return mGlobal_PIX_NX_Tower; }
  int getNumberOfPIXsInY() const { return mGlobal_PIX_NY_Tower; }
  float getHCALTowerSize() const { return mGlobal_HCAL_Tower_Size; }
  int getHCALTowersInX() const { return mGlobal_HCAL_Tower_NX; }
  int getHCALTowersInY() const { return mGlobal_HCAL_Tower_NY; }
  int getNumberOfSegments() const { return mNumberOfSegments; } // NOTE: These are not the virtual segments, but just total number of layers as read from the geometry file. Need to disambiguate
  int getNumberOfPadLayers() const { return mNPadLayers; }
  int getNumberOfPixelLayers() const { return mNPixelLayers; }
  int getNumberOfHCalLayers() const { return mNHCalLayers; }
  int getNumberOfLayers() const { return mNPadLayers + mNPixelLayers + mNHCalLayers; }
  int getNumberOfLayerSeg() const { return mLayerSeg; }

  double getFOCALSizeX() const;
  double getFOCALSizeY() const;
  double getTowerSize() const { return mTowerSizeX; }
  double getTowerSizeX() const;
  double getTowerSizeY() const;
  double getFOCALSizeZ() const;
  double getECALSizeZ() const;
  double getECALCenterZ() const;
  double getHCALSizeZ() const;
  double getHCALCenterZ() const;
  double getFOCALSegmentZ(int seg) const;
  double getFOCALZ0() const { return mGlobal_FOCAL_Z0; }
  int getNumberOfTowersInX() const { return mGlobal_Tower_NX; }
  int getNumberOfTowersInY() const { return mGlobal_Tower_NY; }
  double getTowerGapSize() const { return mGlobal_TOWER_TOLX; }
  double getTowerGapSizeX() const { return mGlobal_TOWER_TOLX; }
  double getTowerGapSizeY() const { return mGlobal_TOWER_TOLY; }
  double getGlobalPixelSize() const { return mGlobal_Pixel_Size; }
  double getGlobalPixelWaferSizeX() const { return mGlobal_PIX_SizeX; }
  double getGlobalPixelWaferSizeY() const { return mGlobal_PIX_SizeY; }
  double getGlobalPixelSkin() const { return mGlobal_PIX_SKIN; }
  double getGlobalPixelOffsetX() const { return mGlobal_PIX_OffsetX; }
  double getGlobalPadSize() const { return mGlobal_Pad_Size; }
  float getMiddleTowerOffset() const { return mGlobal_Middle_Tower_Offset; }
  bool getInsertFrontPadLayers() const { return mInsertFrontPadLayers; }
  bool getInsertHCalReadoutMaterial() const { return mInsertFrontHCalReadoutMaterial; }

  // TObjArray* getFOCALMicroModule(int layer) const;   // NOTE: Check if needed, otherwise remove
  const Composition* getComposition(int layer, int stack) const;
  std::string_view getTowerGapMaterial() const { return mGlobal_Gap_Material; }

  int getVirtualNSegments() const;

  float getVirtualPadSize(int segment) const;
  float getVirtualRelativeSensitiveThickness(int segment) const;
  float getVirtualPixelTreshold(int segment) const;
  float getVirtualSegmentSizeZ(int segment) const;
  float getVirtualSegmentZ(int segment) const;
  bool getVirtualIsPixel(int segment) const;
  bool getVirtualIsHCal(int segment) const;
  int getVirtualNLayersInSegment(int segment) const;
  int getVirtualMinLayerInSegment(int segment) const;
  int getVirtualMaxLayerInSegment(int segment) const;

  void setUpLayerSegmentMap();
  void setUpTowerWaferSize();

  bool getUseHCALSandwich() { return mUseSandwichHCAL; }

 protected:
  std::vector<Composition> mGeometryComposition;
  std::vector<Composition> mFrontMatterCompositionBase;
  std::vector<Composition> mPadCompositionBase;
  std::vector<Composition> mPixelCompositionBase;
  std::vector<Composition> mHCalCompositionBase;

  // PAD setup
  float mGlobal_Pad_Size = 0.0; // pad size
  int mGlobal_PAD_NX = 0;       // number of X pads in wafer
  int mGlobal_PAD_NY = 0;       // number of Y pads in wafer
  int mGlobal_PAD_NX_Tower = 0; // number of X wafers in tower
  int mGlobal_PAD_NY_Tower = 0; // number of Y wafers in tower
  float mGlobal_PPTOL = 0.0;    // tolerance between the wafers
  float mGlobal_PAD_SKIN = 0.0; // dead area (guard ring) on the wafer
  float mWaferSizeX = 0.0;      // Wafer X size
  float mWaferSizeY = 0.0;      // Wafer Y size

  // PIX setup
  float mGlobal_Pixel_Size = 0.0;  // pixel size
  float mGlobal_PIX_SizeX = 0.0;   // sensor size X
  float mGlobal_PIX_SizeY = 0.0;   // sensor size Y
  float mGlobal_PIX_OffsetX = 0.0; // offset for pixel layers in X
  float mGlobal_PIX_OffsetY = 0.0; // offset for pixel layers in Y
  float mGlobal_PIX_SKIN = 0.0;
  int mGlobal_PIX_NX_Tower = 0;       // number of sensors in X
  int mGlobal_PIX_NY_Tower = 0;       // number of sensors in Y
  bool mGlobal_Pixel_Readout = false; // readout on

  // Tower setup
  int mNPadLayers = 0;                      // total number of pad layers
  int mNPixelLayers = 0;                    // number of pixel layers
  std::array<int, 20> mPixelLayerLocations; // location of the pixel layers
  int mGlobal_Tower_NX = 0;                 // How many towers in X
  int mGlobal_Tower_NY = 0;                 // How many towers in Y
  float mTowerSizeX = 0.0;                  // X size of tower
  float mTowerSizeY = 0.0;                  // Y size of tower
  float mGlobal_TOWER_TOLX = 0.0;           // X - tolarance around tower
  float mGlobal_TOWER_TOLY = 0.0;           // Y - tolarance around tower
  float mGlobal_Middle_Tower_Offset = 0.0;  // if odd layers, the middle tower is offset due to the beampipe
  std::string mGlobal_Gap_Material;         // gap filling material	NOTE: currently not used

  float mGlobal_HCAL_Tower_Size = 0.0;
  int mGlobal_HCAL_Tower_NX = 0; // Number of HCAL towers on X
  int mGlobal_HCAL_Tower_NY = 0; // Number of HCAL towers on Y
  bool mUseSandwichHCAL = false;

  float mGlobal_FOCAL_Z0 = 0.0;

  bool mInsertFrontPadLayers = false;           // Have 2 pad layers in front of ECAL for charged particle veto
  bool mInsertFrontHCalReadoutMaterial = false; // if true, insert an 1cm thick aluminium layer at 2cm behind HCal to simulate the material introduced by the readout

  int mLayerSeg = 0;
  int mNHCalLayers = 0;                           // number of HCalLayers
  std::array<int, 100> mSegments;                 //  which layer belongs to which segment
  std::array<int, 100> mNumberOfLayersInSegments; // nymber of layers in each segment
  int mNumberOfSegments = 0;                      // number of long. segements
  int mNFrontMatterCompositionBase = 0;
  std::array<float, 100> mLocalLayerZ;    //// layer location in z
  std::array<float, 100> mLocalSegmentsZ; /// segment location in z
  float mFrontMatterLayerThickness = 0.0;
  float mPadLayerThickness = 0.0;
  float mPixelLayerThickness = 0.0;
  float mHCalLayerThickness = 0.0;
  std::array<float, 100> mLayerThickness; // thickenss of the layers
  std::list<int> mDisableTowers;

  int mVirtualNSegments = 0;
  std::vector<VirtualSegment> mVirtualSegmentComposition;

 private:
  static Geometry* sGeom;
  static bool sInit;
}; // Geometry

} // namespace focal
} // namespace o2
#endif