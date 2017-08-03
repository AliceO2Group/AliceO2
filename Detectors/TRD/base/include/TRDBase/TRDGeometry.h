// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDGEOMETRY_H
#define O2_TRDGEOMETRY_H

#include "TRDBase/TRDGeometryBase.h"
#include "DetectorsCommonDataFormats/DetID.h"

#include <string>
#include <vector>
#include <memory>

class TGeoHMatrix;

namespace o2
{
namespace trd
{
class TRDPadPlane;

class TRDGeometry : public TRDGeometryBase
{
 public:
  TRDGeometry();
  ~TRDGeometry();

  void CreateGeometry(std::vector<int> const& idtmed);
  void CreateVolumes(std::vector<int> const& idtmed);
  static void CreatePadPlaneArray();
  static void CreatePadPlane(int ilayer, int istack, TRDPadPlane& plane);
  void AssembleChamber(int ilayer, int istack);
  void CreateFrame(std::vector<int> const& idtmed);
  void CreateServices(std::vector<int> const& idtmed);
  bool RotateBack(int det, const double* const loc, double* glb) const;

  std::vector<std::string> const& getSensitiveTRDVolumes() const { return mSensitiveVolumeNames; }

 protected:
  static TObjArray* fgClusterMatrixArray; //! Transformation matrices loc. cluster to tracking cs
  static std::unique_ptr<TRDPadPlane[]> fgPadPlaneArray;

 private:
  std::vector<std::string> mSensitiveVolumeNames; //!< vector keeping track of sensitive TRD volumes
  static const o2::detectors::DetID sDetID;

  // helper function to create volumes and registering them automatically
  void createVolume(const char* name, const char* shape, int nmed, float* upar, int np);

  ClassDefNV(TRDGeometry, 1) //  TRD geometry class
};
} // end namespace trd
} // end namespace o2
#endif
