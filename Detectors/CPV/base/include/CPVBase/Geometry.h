// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_GEOMETRY_H_
#define ALICEO2_CPV_GEOMETRY_H_

#include <string>

#include <RStringView.h>
#include <TMath.h>

namespace o2
{
namespace cpv
{
class Geometry
{
 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  Geometry() = default;

  ///
  /// Constructor for normal use.
  ///
  /// \param name: geometry name: CPV (see main class description for definition)
  Geometry(const std::string_view name);

  ///
  /// Copy constructor.
  ///
  Geometry(const Geometry& geom);

  ///
  /// Destructor.
  ///
  ~Geometry() = default;

  ///
  /// Assign operator.
  ///
  Geometry& operator=(const Geometry& rvalue);

  ///
  /// \return the pointer of the _existing_ unique instance of the geometry
  /// It should have been set before with GetInstance(name) method
  ///
  static Geometry* GetInstance()
  {
    if (sGeom) {
      return sGeom;
    } else {
      return GetInstance("Run3CPV");
    }
  }

  ///
  /// \return (newly created) pointer of the unique instance of the geometry. Previous instance is destroied.
  ///
  /// \param name: geometry name: CPV (see main class description for definition)
  /// \param title
  //  \param mcname: Geant3/4, Fluka, needed for settings of transport (check) \param mctitle: Geant4 physics list
  //  (check)
  ///
  static Geometry* GetInstance(const std::string_view name)
  {
    if (sGeom) {
      if (sGeom->GetName() == name) {
        return sGeom;
      } else {
        delete sGeom;
      }
    }
    sGeom = new Geometry(name);
    return sGeom;
  }

  int AreNeighbours(int absId1, int absId2) const;

  ///
  /// \return AbsId index of the CPV cell
  ///
  /// \param moduleNumber: module number
  /// \param strip: strip number
  //  \param cell: cell in strip number
  ///
  int RelToAbsId(int moduleNumber, int iphi, int iz) const;
  bool AbsToRelNumbering(int absId, int* relid) const;
  int AbsIdToModule(int absId);
  void AbsIdToRelPosInModule(int absId, double& x, double& z) const;
  bool RelToAbsNumbering(const int* RelId, int& AbsId) const;
  // converts the absolute CPV numbering to a relative

  int GetTotalNPads() const { return mNumberOfCPVPadsPhi * mNumberOfCPVPadsZ * 3; } // TODO: evaluate from real geometry
  int IsPadExists(int absId) const
  {
    return absId > 0 && absId <= GetTotalNPads();
  } // TODO: evaluate from real geometry

  const std::string& GetName() const { return mGeoName; }

 private:
  static Geometry* sGeom; // Pointer to the unique instance of the singleton

  int mNumberOfCPVPadsPhi; // Number of pads in phi direction
  int mNumberOfCPVPadsZ;   // Number of pads in z direction
  double mCPVPadSizePhi;   // pad size in phi direction (in cm)
  double mCPVPadSizeZ;     // pad size in z direction (in cm)

  std::string mGeoName; ///< Geometry name string
};
} // namespace cpv
} // namespace o2
#endif
