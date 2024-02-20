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
  class Stave
  {
    class Module
    {
      class Sensor
      {
       public:
        Sensor() = default;
        Sensor(std::string sensorName,
               int layer,
               int stave,
               int module,
               int number,
               float moduleOffset = -59.8f,
               float sensorLength = 49.9f,
               float sensorWidth = 2.5f,
               float sensorThickness = 0.5f,
               float sensorSpacing = 0.2f);
        void createSensor(TGeoVolume* motherVolume);

       private:
        std::string mName;
        float mModuleOffset;
        float mWidth;
        float mLength;
        float mThickness;
        float mSpacing;
        int mLayer;
        int mStave;
        int mModule;
        int mNumber;
      };

     public:
      Module() = default;
      Module(std::string moduleName,
             int layer,
             int stave,
             int number,
             int nBars = 23,
             float zOffset = -500.f,
             float barLength = 49.9f,
             float barSpacing = 0.2f,
             float barWidth = 2.5f,
             float barThickness = 0.5f);
      void createModule(TGeoVolume* motherVolume);

     private:
      std::string mName;
      float mBarSpacing;
      float mBarWidth;
      float mBarLength;
      float mBarThickness;
      float mZOffset;
      int mNBars;
      int mLayer;
      int mStave;
      int mNumber;
      std::vector<Sensor> mSensors;
    };

   public:
    Stave() = default;
    Stave(std::string staveName,
          float radDistance,
          float rotAngle,
          int layer,
          int number,
          float staveLength = 500.f,
          float staveWidth = 50.f,
          float staveThickness = 0.5f,
          int nModulesZ = 10);
    void createStave(TGeoVolume* motherVolume);

   private:
    std::string mName;
    float mRadDistance;
    float mRotAngle;
    float mLength;
    float mWidth;
    float mThickness;
    std::vector<Module> mModules;
    int mLayer;
    int mNumber;
    int mNModulesZ;
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