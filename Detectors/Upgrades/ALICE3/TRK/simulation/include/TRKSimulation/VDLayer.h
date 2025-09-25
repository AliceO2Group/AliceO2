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

#ifndef ALICEO2_VD_LAYER_H
#define ALICEO2_VD_LAYER_H

#include <string>
#include <Rtypes.h>

class TGeoVolume;
class TGeoMatrix;

namespace o2
{
namespace trk
{

// Base class for a VD layer
class VDLayer
{
 public:
  VDLayer() = default;
  VDLayer(int layerNumber, const std::string& layerName, double layerX2X0);
  virtual ~VDLayer() = default;

  // Create the layer (AIR container + sensors) and insert it into mother
  virtual void createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans = nullptr) const = 0;

  double getChipThickness() const { return mChipThickness; }

 protected:
  int mLayerNumber{0};
  std::string mLayerName;
  double mX2X0{0.f};          // Radiation length in units of X0
  double mChipThickness{0.f}; // thickness derived from X/X0
  double mModuleWidth{4.54f}; // cm

  // ClassDef(VDLayer, 1)
};

// Cylindrical segment layer
class VDCylindricalLayer : public VDLayer
{
 public:
  VDCylindricalLayer(int layerNumber, const std::string& layerName, double layerX2X0,
                     double radius, double phiSpanDeg, double lengthZ, double lengthSensZ);

  TGeoVolume* createSensor() const; // builds the sensor volume
  void createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans = nullptr) const override;

 private:
  double mRadius{0.f};
  double mPhiSpanDeg{0.f};  // degrees
  double mLengthZ{0.f};     // layer container length in Z
  double mLengthSensZ{0.f}; // sensor length in Z

  // ClassDef(VDCylindricalLayer, 1)
};

// Rectangular segment layer
class VDRectangularLayer : public VDLayer
{
 public:
  VDRectangularLayer(int layerNumber, const std::string& layerName, double layerX2X0,
                     double width, double lengthZ, double lengthSensZ);

  TGeoVolume* createSensor() const;
  void createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans = nullptr) const override;

 private:
  double mWidth{0.f};
  double mLengthZ{0.f};
  double mLengthSensZ{0.f};

  // ClassDef(VDRectangularLayer, 1)
};

// Disk segment layer
class VDDiskLayer : public VDLayer
{
 public:
  VDDiskLayer(int layerNumber, const std::string& layerName, double layerX2X0,
              double rMin, double rMax, double phiSpanDeg, double zPos);

  TGeoVolume* createSensor() const;
  void createLayer(TGeoVolume* motherVolume, TGeoMatrix* combiTrans = nullptr) const override;

  double getZPosition() const { return mZPos; }

 private:
  double mRMin{0.f};
  double mRMax{0.f};
  double mPhiSpanDeg{0.f}; // degrees
  double mZPos{0.f};       // placement along Z

  // ClassDef(VDDiskLayer, 1)
};

} // namespace trk
} // namespace o2

#endif // ALICEO2_VD_LAYER_H
