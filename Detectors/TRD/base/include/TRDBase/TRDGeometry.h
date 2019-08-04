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
  ~TRDGeometry() override = default;

  void createGeometry(std::vector<int> const& idtmed);
  void addAlignableVolumes() const;
  bool createClusterMatrixArray();
  void createPadPlaneArray();

  bool rotateBack(int det, const float* const loc, float* glb) const;
  bool chamberInGeometry(int det) const;
  std::vector<std::string> const& getSensitiveTRDVolumes() const { return mSensitiveVolumeNames; }

 protected:
  void fillMatrixCache(int mask) override;

 private:
  void createVolumes(std::vector<int> const& idtmed);
  void assembleChamber(int ilayer, int istack);
  void createFrame(std::vector<int> const& idtmed);
  void createServices(std::vector<int> const& idtmed);
  void createPadPlane(int ilayer, int istack);

  std::vector<std::string> mSensitiveVolumeNames; //!< vector keeping track of sensitive TRD volumes
  static const o2::detectors::DetID sDetID;

  // helper function to create volumes and registering them automatically
  void createVolume(const char* name, const char* shape, int nmed, float* upar, int np);

  ClassDefOverride(TRDGeometry, 1); //  TRD geometry class
};
} // end namespace trd
} // end namespace o2
#endif
