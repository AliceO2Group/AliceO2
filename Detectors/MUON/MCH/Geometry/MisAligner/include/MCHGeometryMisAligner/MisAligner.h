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

/** @file MisAligner.h
 * Generate misalignments.
 * @author Javier Castillo Castellanos, Aude Glaenzer
 */

#ifndef O2_MCH_GEOMETRY_MIS_ALIGNER
#define O2_MCH_GEOMETRY_MIS_ALIGNER

#include "TObject.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include <vector>

class TGeoCombiTrans;
class TClonesArray;

namespace o2
{
namespace mch
{
namespace geo
{

class MisAligner : public TObject
{
 public:
  MisAligner(double cartXMisAligM, double cartXMisAligW, double cartYMisAligM, double cartYMisAligW, double angMisAligM, double angMisAligW);

  MisAligner(double cartMisAligM, double cartMisAligW, double angMisAligM, double angMisAligW);

  MisAligner(double cartMisAligW, double angMisAligW);

  MisAligner();

  ~MisAligner() final = default;

  //_________________________________________________________________
  // methods

  // misaligns the geometry and fills an array of misalignment paramaters.
  void misAlign(std::vector<o2::detectors::AlignParam>& arr, bool verbose = false) const;

  /// Set cartesian displacement parameters different along x, y
  void setCartMisAlig(double xmean, double xwidth, double ymean, double ywidth, double zmean = 0., double zwidth = 0.)
  {
    mDetElemMisAlig[0][0] = xmean;
    mDetElemMisAlig[0][1] = xwidth;
    mDetElemMisAlig[1][0] = ymean;
    mDetElemMisAlig[1][1] = ywidth;
    mDetElemMisAlig[2][0] = zmean;
    mDetElemMisAlig[2][1] = zwidth;
  }

  /// Set cartesian displacement parameters, the same along x, y
  void setCartMisAlig(double mean, double width)
  {
    mDetElemMisAlig[0][0] = mean;
    mDetElemMisAlig[0][1] = width;
    mDetElemMisAlig[1][0] = mean;
    mDetElemMisAlig[1][1] = width;
  }

  /// Set angular displacement
  void setAngMisAlig(double zmean, double zwidth, double xmean = 0., double xwidth = 0., double ymean = 0., double ywidth = 0.)
  {
    mDetElemMisAlig[3][0] = xmean;
    mDetElemMisAlig[3][1] = xwidth;
    mDetElemMisAlig[4][0] = ymean;
    mDetElemMisAlig[4][1] = ywidth;
    mDetElemMisAlig[5][0] = zmean;
    mDetElemMisAlig[5][1] = zwidth;
  }

  void setXYAngMisAligFactor(double factor);

  void setZCartMisAligFactor(double factor);

  /// Set option for gaussian distribution
  void setUseGaus(bool usegaus)
  {
    mUseGaus = usegaus;
    mUseUni = !usegaus;
  }

  /// Set option for uniform distribution
  void setUseUni(bool useuni)
  {
    mUseGaus = !useuni;
    mUseUni = useuni;
  }

  /// Set module (half chambers) cartesian displacement parameters
  void setModuleCartMisAlig(double xmean, double xwidth, double ymean, double ywidth, double zmean, double zwidth)
  {
    mModuleMisAlig[0][0] = xmean;
    mModuleMisAlig[0][1] = xwidth;
    mModuleMisAlig[1][0] = ymean;
    mModuleMisAlig[1][1] = ywidth;
    mModuleMisAlig[2][0] = zmean;
    mModuleMisAlig[2][1] = zwidth;
  }

  /// Set module (half chambers) cartesian displacement parameters
  void setModuleAngMisAlig(double xmean, double xwidth, double ymean, double ywidth, double zmean, double zwidth)
  {
    mModuleMisAlig[3][0] = xmean;
    mModuleMisAlig[3][1] = xwidth;
    mModuleMisAlig[4][0] = ymean;
    mModuleMisAlig[4][1] = ywidth;
    mModuleMisAlig[5][0] = zmean;
    mModuleMisAlig[5][1] = zwidth;
  }

  /// Set alignment resolution to misalign objects to be stored in CDB
  void setAlignmentResolution(const TClonesArray* misAlignArray, int chId = -1, double chResX = -1., double chResY = -1., double deResX = -1., double deResY = -1.);

 protected:
  /// Not implemented
  MisAligner(const MisAligner& right);
  /// Not implemented
  MisAligner& operator=(const MisAligner& right);

 private:
  bool isMatrixConvertedToAngles(const double* rot, double& psi, double& theta, double& phi) const;
  // return a misaligned transformation
  TGeoCombiTrans misAlignDetElem() const;
  TGeoCombiTrans misAlignModule() const;
  void getUniMisAlign(double cartMisAlig[3], double angMisAlig[3], const double lParMisAlig[6][2]) const;
  void getGausMisAlign(double cartMisAlig[3], double angMisAlig[3], const double lParMisAlig[6][2]) const;

  bool mUseUni;                 ///< use uniform distribution for misaligmnets
  bool mUseGaus;                ///< use gaussian distribution for misaligmnets
  double mDetElemMisAlig[6][2]; ///< Mean and width of the displacements of the detection elements along x,y,z (translations) and about x,y,z (rotations)
  double mModuleMisAlig[6][2];  ///< Mean and width of the displacements of the modules along x,y,z (translations) and about x,y,z (rotations)

  double mXYAngMisAligFactor; ///< factor (<1) to apply to angular misalignment range since range of motion is restricted out of the xy plane
  double mZCartMisAligFactor; ///< factor (<1) to apply to cartetian misalignment range since range of motion is restricted in z direction

  ClassDef(MisAligner, 1);
};

} // namespace geo
} // namespace mch
} // namespace o2
#endif // GEOMETRY_MIS_ALIGNER_H
