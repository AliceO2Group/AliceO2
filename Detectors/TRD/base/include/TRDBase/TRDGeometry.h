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
#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "DetectorsCommonDataFormats/DetID.h"

#include <string>
#include <vector>
#include <memory>

using namespace o2::trd;

namespace o2
{
namespace trd
{

class TRDGeometry : public TRDGeometryBase, public o2::detectors::DetMatrixCacheIndirect
{
 public:
  TRDGeometry();
  virtual ~TRDGeometry() override = default;

  void CreateGeometry(std::vector<int> const& idtmed);
  void addAlignableVolumes() const;
  bool createClusterMatrixArray();

  bool RotateBack(int det, const double* const loc, double* glb) const;
  bool ChamberInGeometry(int det);
  const Mat3D* GetClusterMatrix(int det);

  std::vector<std::string> const& getSensitiveTRDVolumes() const { return mSensitiveVolumeNames; }

 protected:
  virtual void fillMatrixCache(int mask) override;

  static std::unique_ptr<TRDPadPlane[]> fgPadPlaneArray;

 private:
  void CreateVolumes(std::vector<int> const& idtmed);
  void AssembleChamber(int ilayer, int istack);
  void CreateFrame(std::vector<int> const& idtmed);
  void CreateServices(std::vector<int> const& idtmed);
  static void CreatePadPlaneArray();
  static void CreatePadPlane(int ilayer, int istack);

  std::vector<std::string> mSensitiveVolumeNames; //!< vector keeping track of sensitive TRD volumes
  static const o2::detectors::DetID sDetID;

  // helper function to create volumes and registering them automatically
  void createVolume(const char* name, const char* shape, int nmed, float* upar, int np);

  ClassDefOverride(TRDGeometry, 1) //  TRD geometry class
};
} // end namespace trd
} // end namespace o2
#endif
