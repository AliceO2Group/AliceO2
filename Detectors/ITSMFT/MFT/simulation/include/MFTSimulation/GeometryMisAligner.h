// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryMisAligner.h
/// \brief This macro is used to misalign the existing MFT geometry
/// \author robin.caron@cern.ch (based on MUON/MCH AliRoot macros)
/// \date 01/07/2020
///

#ifndef ALICEO2_MFT_GEOMETRYMISALIGNER_H
#define ALICEO2_MFT_GEOMETRYMISALIGNER_H

#include <TObject.h>

class TGeoCombiTrans;
class TClonesArray;

namespace o2
{
namespace mft
{
class GeometryTGeo;
}
} // namespace o2

namespace o2
{
namespace mft
{
class GeometryMisAligner
{
 public:
  GeometryMisAligner(Double_t cartXMisAligM, Double_t cartXMisAligW, Double_t cartYMisAligM, Double_t cartYMisAligW, Double_t angMisAligM, Double_t angMisAligW);
  GeometryMisAligner(Double_t cartMisAligM, Double_t cartMisAligW, Double_t angMisAligM, Double_t angMisAligW);
  GeometryMisAligner(Double_t cartMisAligW, Double_t angMisAligW);
  GeometryMisAligner();
  ~GeometryMisAligner() = default;

  /// Not implemented
  GeometryMisAligner(const GeometryMisAligner& right);
  /// Not implemented
  GeometryMisAligner& operator=(const GeometryMisAligner& right);

  //_________________________________________________________________
  // methods

  GeometryTGeo* mGeometryTGeo; //! access to geometry details

  bool matrixToAngles(const double* rot, double& psi, double& theta, double& phi);

  // return a misaligned geometry obtained from the existing one.
  void MisAlign(bool verbose = false,
                const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080",
                long tmin = 0, long tmax = -1,
                const std::string& objectPath = "",
                const std::string& fileName = "MFTAlignment.root");

  /// Set sensor cartesian displacement parameters different along x, y
  void SetSensorCartMisAlig(Double_t xmean, Double_t xwidth, Double_t ymean, Double_t ywidth, Double_t zmean = 0., Double_t zwidth = 0.)
  {
    fSensorMisAlig[0][0] = xmean;
    fSensorMisAlig[0][1] = xwidth;
    fSensorMisAlig[1][0] = ymean;
    fSensorMisAlig[1][1] = ywidth;
    fSensorMisAlig[2][0] = zmean;
    fSensorMisAlig[2][1] = zwidth;
  }

  /// Set sensor cartesian displacement parameters, the same along x, y
  void SetSensorCartMisAlig(Double_t mean, Double_t width)
  {
    fSensorMisAlig[0][0] = mean;
    fSensorMisAlig[0][1] = width;
    fSensorMisAlig[1][0] = mean;
    fSensorMisAlig[1][1] = width;
  }

  /// Set sensor angular displacement
  void SetSensorAngMisAlig(Double_t zmean, Double_t zwidth, Double_t xmean = 0., Double_t xwidth = 0., Double_t ymean = 0., Double_t ywidth = 0.)
  {
    fSensorMisAlig[3][0] = xmean;
    fSensorMisAlig[3][1] = xwidth;
    fSensorMisAlig[4][0] = ymean;
    fSensorMisAlig[4][1] = ywidth;
    fSensorMisAlig[5][0] = zmean;
    fSensorMisAlig[5][1] = zwidth;
  }

  /// Set sensor cartesian displacement (Kept for backward compatibility)
  void SetMaxSensorCartMisAlig(Double_t width)
  {
    fSensorMisAlig[0][0] = 0.0;
    fSensorMisAlig[0][1] = width;
    fSensorMisAlig[1][0] = 0.0;
    fSensorMisAlig[1][1] = width;
  }

  /// Set sensor angular displacement (Kept for backward compatibility)
  void SetMaxSensorAngMisAlig(Double_t width)
  {
    fSensorMisAlig[5][0] = 0.0;
    fSensorMisAlig[5][1] = width;
  }

  /// Set sensor cartesian displacement parameters different along x, y
  void SetLadderCartMisAlig(Double_t xmean, Double_t xwidth, Double_t ymean, Double_t ywidth, Double_t zmean = 0., Double_t zwidth = 0.)
  {
    fLadderMisAlig[0][0] = xmean;
    fLadderMisAlig[0][1] = xwidth;
    fLadderMisAlig[1][0] = ymean;
    fLadderMisAlig[1][1] = ywidth;
    fLadderMisAlig[2][0] = zmean;
    fLadderMisAlig[2][1] = zwidth;
  }

  /// Set ladder cartesian displacement parameters, the same along x, y
  void SetLadderCartMisAlig(Double_t mean, Double_t width)
  {
    fLadderMisAlig[0][0] = mean;
    fLadderMisAlig[0][1] = width;
    fLadderMisAlig[1][0] = mean;
    fLadderMisAlig[1][1] = width;
  }

  /// Set ladder angular displacement
  void SetLadderAngMisAlig(Double_t zmean, Double_t zwidth, Double_t xmean = 0., Double_t xwidth = 0., Double_t ymean = 0., Double_t ywidth = 0.)
  {
    fLadderMisAlig[3][0] = xmean;
    fLadderMisAlig[3][1] = xwidth;
    fLadderMisAlig[4][0] = ymean;
    fLadderMisAlig[4][1] = ywidth;
    fLadderMisAlig[5][0] = zmean;
    fLadderMisAlig[5][1] = zwidth;
  }

  /// Set cartesian displacement for ladder (Kept for backward compatibility)
  void SetMaxLadderCartMisAlig(Double_t width)
  {
    fLadderMisAlig[0][0] = 0.0;
    fLadderMisAlig[0][1] = width;
    fLadderMisAlig[1][0] = 0.0;
    fLadderMisAlig[1][1] = width;
  }

  /// Set angular displacement for ladder (Kept for backward compatibility)
  void SetMaxLadderAngMisAlig(Double_t width)
  {
    fLadderMisAlig[5][0] = 0.0;
    fLadderMisAlig[5][1] = width;
  }

  void SetXYAngMisAligFactor(Double_t factor);

  void SetZCartMisAligFactor(Double_t factor);

  /// Set option for gaussian distribution
  void SetUseGaus(Bool_t usegaus)
  {
    fUseGaus = usegaus;
    fUseUni = !usegaus;
  }

  /// Set option for uniform distribution
  void SetUseUni(Bool_t useuni)
  {
    fUseGaus = !useuni;
    fUseUni = useuni;
  }

  /// Set half cartesian displacement parameters
  void SetHalfCartMisAlig(Double_t xmean, Double_t xwidth, Double_t ymean, Double_t ywidth, Double_t zmean, Double_t zwidth)
  {
    fHalfMisAlig[0][0] = xmean;
    fHalfMisAlig[0][1] = xwidth;
    fHalfMisAlig[1][0] = ymean;
    fHalfMisAlig[1][1] = ywidth;
    fHalfMisAlig[2][0] = zmean;
    fHalfMisAlig[2][1] = zwidth;
  }

  /// Set half cartesian displacement parameters
  void SetHalfAngMisAlig(Double_t xmean, Double_t xwidth, Double_t ymean, Double_t ywidth, Double_t zmean, Double_t zwidth)
  {
    fHalfMisAlig[3][0] = xmean;
    fHalfMisAlig[3][1] = xwidth;
    fHalfMisAlig[4][0] = ymean;
    fHalfMisAlig[4][1] = ywidth;
    fHalfMisAlig[5][0] = zmean;
    fHalfMisAlig[5][1] = zwidth;
  }

  /// Set disk cartesian displacement parameters
  void SetDiskCartMisAlig(Double_t xmean, Double_t xwidth, Double_t ymean, Double_t ywidth, Double_t zmean, Double_t zwidth)
  {
    fDiskMisAlig[0][0] = xmean;
    fDiskMisAlig[0][1] = xwidth;
    fDiskMisAlig[1][0] = ymean;
    fDiskMisAlig[1][1] = ywidth;
    fDiskMisAlig[2][0] = zmean;
    fDiskMisAlig[2][1] = zwidth;
  }

  /// Set disk cartesian displacement parameters
  void SetDiskAngMisAlig(Double_t xmean, Double_t xwidth, Double_t ymean, Double_t ywidth, Double_t zmean, Double_t zwidth)
  {
    fDiskMisAlig[3][0] = xmean;
    fDiskMisAlig[3][1] = xwidth;
    fDiskMisAlig[4][0] = ymean;
    fDiskMisAlig[4][1] = ywidth;
    fDiskMisAlig[5][0] = zmean;
    fDiskMisAlig[5][1] = zwidth;
  }

  /// Set alignment resolution to misalign objects to be stored in CCDB
  void SetAlignmentResolution(const TClonesArray* misAlignArray, Int_t chId = -1, Double_t chResX = -1., Double_t chResY = -1., Double_t deResX = -1., Double_t deResY = -1.);

 protected:
 private:
  // return a misaligned transformation
  TGeoCombiTrans MisAlignLadder() const;
  TGeoCombiTrans MisAlignDisk() const;
  TGeoCombiTrans MisAlignSensor() const;
  TGeoCombiTrans MisAlignHalf() const;

  void GetUniMisAlign(Double_t cartMisAlig[3], Double_t angMisAlig[3], const Double_t lParMisAlig[6][2]) const;
  void GetGausMisAlign(Double_t cartMisAlig[3], Double_t angMisAlig[3], const Double_t lParMisAlig[6][2]) const;

  Bool_t fUseUni;  ///< use uniform distribution for misaligmnets
  Bool_t fUseGaus; ///< use gaussian distribution for misaligmnets

  Double_t fSensorMisAlig[6][2]; ///< Mean and width of the displacements of the sensors along x,y,z (translations) and about x,y,z (rotations)
  Double_t fLadderMisAlig[6][2]; ///< Mean and width of the displacements of the ladder along x,y,z (translations) and about x,y,z (rotations)
  Double_t fDiskMisAlig[6][2];   ///< Mean and width of the displacements of the half-disk along x,y,z (translations) and about x,y,z (rotations)
  Double_t fHalfMisAlig[6][2];   ///< Mean and width of the displacements of the half-MF along x,y,z (translations) and about x,y,z (rotations)

  Double_t fXYAngMisAligFactor; ///< factor (<1) to apply to angular misalignment range since range of motion is restricted out of the xy plane
  Double_t fZCartMisAligFactor; ///< factor (<1) to apply to cartetian misalignment range since range of motion is restricted in z direction

  ClassDef(GeometryMisAligner, 0) // Geometry parametrisation
};

} // namespace mft
} // namespace o2

#endif //ALICEO2_MFT_GEOMETRYMISALIGNER_H
