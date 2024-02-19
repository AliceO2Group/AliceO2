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

#ifndef MI3LAYER_H_
#define MI3LAYER_H_

#include <string>
#include <vector>

class TGeoVolume;
class TGeoVolumeAssembly;
namespace o2::mi3
{
class MIDLayer
{
  friend class Stave;
  class Stave
  {
    class Module
    {
     public:
      Module(std::string moduleName,
             int nBars,
             float barSpacing,
             float barWidth,
             float barLength,
             float barThickness);

     private:
      std::string mName;
    };

   public:
    Stave() = default;
    Stave(std::string staveName,
          float radDistance,
          float rotAngle,
          int layer,
          float staveLength = 500.f,
          float staveWidth = 50.f,
          float staveThickness = 0.5f);
    void createStave(TGeoVolume* motherVolume);

   private:
    std::string mName;
    float mRadDistance;
    float mRotAngle;
    float mLength;
    float mWidth;
    float mThickness;
    std::vector<Module> mModules;
    TGeoVolume* mStaveVolume;
    int mLayer;
  };

 public:
  MIDLayer() = default;
  MIDLayer(int layerNumber, std::string layerName, float rInn, float length, int nstaves = 16);
  void createLayer(TGeoVolume* motherVolume);

 private:
  std::string mName;
  std::vector<Stave> mStaves;
  float mRadius;
  float mLength;
  int mNumber;
  int mNStaves;
};
} // namespace o2::mi3

#endif // MI3LAYER_H