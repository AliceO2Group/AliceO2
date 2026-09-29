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

#ifndef ALICEO2_CPV_GEOMETRYPARAMS_H_
#define ALICEO2_CPV_GEOMETRYPARAMS_H_

#include <string>

#include <RStringView.h>
#include <TNamed.h>
//#include <TVector3.h>

namespace o2
{
namespace cpv
{
class GeometryParams final : public TNamed
{
 public:
  /// Default constructor
  GeometryParams() = default;

  /// Destructor
  ~GeometryParams() final = default;

  /// Get singleton (create if necessary)
  static GeometryParams* GetInstance(const std::string_view name = "CPVRun3Params")
  {
    if (!sGeomParam) {
      sGeomParam = new GeometryParams(name);
    }
    return sGeomParam;
  }

  void GetModuleAngle(int module, double angle[3][2]) const
  {
    for (int i = 0; i < 3; i++) {
      for (int ian = 0; ian < 2; ian++) {
        angle[i][ian] = mModuleAngle[module][i][ian];
      }
    }
  }

  double GetCPVAngle(Int_t index) const { return mCPVAngle[index - 1]; }

  void GetModuleCenter(int module, double* pos) const
  {
    for (int i = 0; i < 3; i++) {
      pos[i] = mModuleCenter[module][i];
    }
  }

  int GetNModules() const { return mNModules; }
  int GetNumberOfCPVPadsPhi() const { return mNumberOfCPVPadsPhi; }
  int GetNumberOfCPVPadsZ() const { return mNumberOfCPVPadsZ; }
  double GetCPVPadSizePhi() const { return mCPVPadSizePhi; }
  double GetCPVPadSizeZ() const { return mCPVPadSizeZ; }
  double GetCPVBoxSize(int index) const { return mCPVBoxSize[index]; }
  double GetCPVActiveSize(int index) const { return mCPVActiveSize[index]; }
  int GetNumberOfCPVChipsPhi() const { return mNumberOfCPVChipsPhi; }
  int GetNumberOfCPVChipsZ() const { return mNumberOfCPVChipsZ; }
  double GetGassiplexChipSize(int index) const { return mGassiplexChipSize[index]; }
  double GetCPVGasThickness() const { return mCPVGasThickness; }
  double GetCPVTextoliteThickness() const { return mCPVTextoliteThickness; }
  double GetCPVCuNiFoilThickness() const { return mCPVCuNiFoilThickness; }
  double GetFTPosition(int index) const { return mFTPosition[index]; }
  double GetCPVFrameSize(int index) const { return mCPVFrameSize[index]; }

 private:
  ///
  /// Main constructor
  ///
  /// Geometry configuration: Run2,...
  GeometryParams(const std::string_view name);

  static GeometryParams* sGeomParam; ///< Pointer to the unique instance of the singleton

  int mNModules;                // Number of CPV modules
  int mNumberOfCPVPadsPhi;      // Number of CPV pads in phi
  int mNumberOfCPVPadsZ;        // Number of CPV pads in z
  double mCPVPadSizePhi;        // CPV pad size in phi
  double mCPVPadSizeZ;          // CPV pad size in z
  double mCPVBoxSize[3];        // Outer size of CPV box
  double mCPVActiveSize[2];     // Active size of CPV box (x,z)
  int mNumberOfCPVChipsPhi;     // Number of CPV Gassiplex chips in phi
  int mNumberOfCPVChipsZ;       // Number of CPV Gassiplex chips in z
  double mGassiplexChipSize[3]; // Size of a Gassiplex chip (0 - in z, 1 - in phi, 2 - thickness (in ALICE radius))
  double mCPVGasThickness;      // Thickness of CPV gas volume
  double mCPVTextoliteThickness; // Thickness of CPV textolite PCB (without moil)
  double mCPVCuNiFoilThickness;  // Thickness of CPV Copper-Nickel moil of PCB
  double mFTPosition[4];         // Positions of the 4 PCB vs the CPV box center
  double mCPVFrameSize[3];       // CPV frame size (0 - in phi, 1 - in z, 2 - thickness (along ALICE radius))
  double mIPtoCPVSurface;        // Distance from IP to CPV front cover
  double mModuleAngle[5][3][2];  // Orientation angles of CPV modules
  double mCPVAngle[5];           // Direction to the center of CPV modules in phi
  double mModuleCenter[5][3];    // Coordunates of modules centra in ALICE system
  ClassDefOverride(GeometryParams, 2);
};
} // namespace cpv
} // namespace o2
#endif
