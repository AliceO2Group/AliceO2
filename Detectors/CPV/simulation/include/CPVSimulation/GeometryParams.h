// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
class GeometryParams : public TNamed
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

  void GetModuleAngle(int module, float angle[3][2]) const
  {
    for (int i = 0; i < 3; i++)
      for (int ian = 0; ian < 2; ian++)
        angle[i][ian] = mModuleAngle[module][i][ian];
  }
  void GetModuleCenter(int module, float* pos) const
  {
    for (int i = 0; i < 3; i++)
      pos[i] = mModuleCenter[module][i];
  }

  int GetNumberOfCPVPadsPhi() const { return mNumberOfCPVPadsPhi; }
  int GetNumberOfCPVPadsZ() const { return mNumberOfCPVPadsZ; }
  float GetCPVPadSizePhi() const { return mCPVPadSizePhi; }
  float GetCPVPadSizeZ() const { return mCPVPadSizeZ; }
  float GetCPVBoxSize(int index) const { return mCPVBoxSize[index]; }
  float GetCPVActiveSize(int index) const { return mCPVActiveSize[index]; }
  int GetNumberOfCPVChipsPhi() const { return mNumberOfCPVChipsPhi; }
  int GetNumberOfCPVChipsZ() const { return mNumberOfCPVChipsZ; }
  float GetGassiplexChipSize(int index) const { return mGassiplexChipSize[index]; }
  float GetCPVGasThickness() const { return mCPVGasThickness; }
  float GetCPVTextoliteThickness() const { return mCPVTextoliteThickness; }
  float GetCPVCuNiFoilThickness() const { return mCPVCuNiFoilThickness; }
  float GetFTPosition(int index) const { return mFTPosition[index]; }
  float GetCPVFrameSize(int index) const { return mCPVFrameSize[index]; }

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
  float mCPVPadSizePhi;         // CPV pad size in phi
  float mCPVPadSizeZ;           // CPV pad size in z
  float mCPVBoxSize[3];         // Outer size of CPV box
  float mCPVActiveSize[2];      // Active size of CPV box (x,z)
  int mNumberOfCPVChipsPhi;     // Number of CPV Gassiplex chips in phi
  int mNumberOfCPVChipsZ;       // Number of CPV Gassiplex chips in z
  float mGassiplexChipSize[3];  // Size of a Gassiplex chip (0 - in z, 1 - in phi, 2 - thickness (in ALICE radius))
  float mCPVGasThickness;       // Thickness of CPV gas volume
  float mCPVTextoliteThickness; // Thickness of CPV textolite PCB (without moil)
  float mCPVCuNiFoilThickness;  // Thickness of CPV Copper-Nickel moil of PCB
  float mFTPosition[4];         // Positions of the 4 PCB vs the CPV box center
  float mCPVFrameSize[3];       // CPV frame size (0 - in phi, 1 - in z, 2 - thickness (along ALICE radius))
  float mIPtoCPVSurface;        // Distance from IP to CPV front cover
  float mModuleAngle[5][3][2];  // Orientation angles of CPV modules
  float mModuleCenter[5][3];    // Coordunates of modules centra in ALICE system
  ClassDefOverride(GeometryParams, 1);
};
} // namespace cpv
} // namespace o2
#endif
