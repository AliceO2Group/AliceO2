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

#ifndef ALICEO2_PASSIVE_PIPERUN4_H
#define ALICEO2_PASSIVE_PIPERUN4_H

#include "DetectorsPassive/PassiveBase.h"
#include "Rtypes.h" // for PipeRun4::Class, ClassDef, PipeRun4::Streamer

class TGeoPcon;

namespace o2
{
namespace passive
{
class PipeRun4 : public PassiveBase
{
 public:
  PipeRun4(const char* name, const char* Title = "Alice Pipe", float rho = 0.f, float thick = 0.f);
  PipeRun4();

  ~PipeRun4() override;
  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

  float getRmin() const { return mBePipeRmax - mBePipeThick; }
  float getRmax() const { return mBePipeRmax; }
  float getWidth() const { return mBePipeThick; }
  float getDz() const { return mIpHLength; }

 private:
  void createMaterials();
  PipeRun4(const PipeRun4& orig);
  PipeRun4& operator=(const PipeRun4&);

  TGeoPcon* makeMotherFromTemplate(const TGeoPcon* shape, int imin = -1, int imax = -1, float r0 = 0.,
                                   int nz = -1);
  TGeoPcon* makeInsulationFromTemplate(TGeoPcon* shape);
  TGeoVolume* makeBellow(const char* ext, int nc, float rMin, float rMax, float rPlie,
                         float dPlie);
  TGeoVolume* makeBellowCside(const char* ext, int nc, float rMin, float rMax, float rPlie, float dPlie);

  TGeoVolume* makeSupportBar(const char* tag, float Rin, float Rout, float length, float skinLength);

  float mBePipeRmax = 0.;  // outer diameter of the Be section
  float mBePipeThick = 0.; // Be section thickness
  float mIpHLength = 0.;   // half length of the beampipe around the IP // FixMe: up to now, hardcoded to 57.25cm

  ClassDefOverride(PipeRun4, 1);
};
} // namespace passive
} // namespace o2
#endif // ALICEO2_PASSIVE_PIPERUN4_H
