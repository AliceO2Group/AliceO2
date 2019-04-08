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
    if (!sGeomParam)
      sGeomParam = new GeometryParams(name);
    return sGeomParam;
  }

  void GetModuleAngle(Int_t module, Float_t angle[3][2]) const
  {
    for (int i = 0; i < 3; i++)
      for (int ian = 0; ian < 2; ian++)
        angle[i][ian] = mModuleAngle[module][i][ian];
  }
  void GetModuleCenter(Int_t module, Float_t* pos) const
  {
    for (int i = 0; i < 3; i++)
      pos[i] = mModuleCenter[module][i];
  }

  Int_t GetNumberOfCPVPadsPhi(void) const { return mNumberOfCPVPadsPhi; }
  Int_t GetNumberOfCPVPadsZ(void) const { return mNumberOfCPVPadsZ; }
  Float_t GetCPVPadSizePhi(void) const { return mCPVPadSizePhi; }
  Float_t GetCPVPadSizeZ(void) const { return mCPVPadSizeZ; }
  Float_t GetCPVBoxSize(Int_t index) const { return mCPVBoxSize[index]; }
  Float_t GetCPVActiveSize(Int_t index) const { return mCPVActiveSize[index]; }
  Int_t GetNumberOfCPVChipsPhi(void) const { return mNumberOfCPVChipsPhi; }
  Int_t GetNumberOfCPVChipsZ(void) const { return mNumberOfCPVChipsZ; }
  Float_t GetGassiplexChipSize(Int_t index) const { return mGassiplexChipSize[index]; }
  Float_t GetCPVGasThickness(void) const { return mCPVGasThickness; }
  Float_t GetCPVTextoliteThickness(void) const { return mCPVTextoliteThickness; }
  Float_t GetCPVCuNiFoilThickness(void) const { return mCPVCuNiFoilThickness; }
  Float_t GetFTPosition(Int_t index) const { return mFTPosition[index]; }
  Float_t GetCPVFrameSize(Int_t index) const { return mCPVFrameSize[index]; }

 private:
  ///
  /// Main constructor
  ///
  /// Geometry configuration: Run2,...
  GeometryParams(const std::string_view name);

  static GeometryParams* sGeomParam; ///< Pointer to the unique instance of the singleton

  Int_t mNModules;                // Number of CPV modules
  Int_t mNumberOfCPVPadsPhi;      // Number of CPV pads in phi
  Int_t mNumberOfCPVPadsZ;        // Number of CPV pads in z
  Float_t mCPVPadSizePhi;         // CPV pad size in phi
  Float_t mCPVPadSizeZ;           // CPV pad size in z
  Float_t mCPVBoxSize[3];         // Outer size of CPV box
  Float_t mCPVActiveSize[2];      // Active size of CPV box (x,z)
  Int_t mNumberOfCPVChipsPhi;     // Number of CPV Gassiplex chips in phi
  Int_t mNumberOfCPVChipsZ;       // Number of CPV Gassiplex chips in z
  Float_t mGassiplexChipSize[3];  // Size of a Gassiplex chip (0 - in z, 1 - in phi, 2 - thickness (in ALICE radius))
  Float_t mCPVGasThickness;       // Thickness of CPV gas volume
  Float_t mCPVTextoliteThickness; // Thickness of CPV textolite PCB (without moil)
  Float_t mCPVCuNiFoilThickness;  // Thickness of CPV Copper-Nickel moil of PCB
  Float_t mFTPosition[4];         // Positions of the 4 PCB vs the CPV box center
  Float_t mCPVFrameSize[3];       // CPV frame size (0 - in phi, 1 - in z, 2 - thickness (along ALICE radius))
  Float_t mIPtoCPVSurface;        // Distance from IP to CPV front cover
  Float_t mModuleAngle[5][3][2];  // Orientation angles of CPV modules
  Float_t mModuleCenter[5][3];    // Coordunates of modules centra in ALICE system
  ClassDefOverride(GeometryParams, 1)
};
} // namespace cpv
} // namespace o2
#endif
