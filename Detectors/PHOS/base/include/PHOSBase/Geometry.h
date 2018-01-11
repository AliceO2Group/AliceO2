// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_GEOMETRY_H_
#define ALICEO2_PHOS_GEOMETRY_H_

#include <string>

#include <RStringView.h>
#include <TMath.h>

namespace o2
{
namespace phos
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
  /// \param name: geometry name: PHOS (see main class description for definition)
  /// \param title \param mcname: Geant3/4, Flukla, needed for settings of transport (check)
  /// \param mctitle: Geant4 physics list (check)
  ///
  Geometry(const std::string_view name, const std::string_view mcname = "", const std::string_view mctitle = "");

  ///
  /// Copy constructor.
  ///
  Geometry(const Geometry& geom);

  ///
  /// Destructor.
  ///
  ~Geometry() {}

  ///
  /// Assign operator.
  ///
  Geometry& operator=(const Geometry& rvalue);

  ///
  /// \return the pointer of the _existing_ unique instance of the geometry
  /// It should have been set before with GetInstance(name) method
  ///
  static Geometry* GetInstance() { return sGeom; }

  ///
  /// \return (newly created) pointer of the unique instance of the geometry. Previous instance is destroied.
  ///
  /// \param name: geometry name: PHOS (see main class description for definition)
  /// \param title
  //  \param mcname: Geant3/4, Fluka, needed for settings of transport (check) \param mctitle: Geant4 physics list
  //  (check)
  ///
  static Geometry* GetInstance(const std::string_view name, const std::string_view mcname = "TGeant3",
                               const std::string_view mctitle = "")
  {
    return sGeom;
  }

  ///
  /// \return AbsId index of the PHOS cell
  ///
  /// \param moduleNumber: module number
  /// \param strip: strip number
  //  \param cell: cell in strip number
  ///
  Int_t RelToAbsId(Int_t moduleNumber, Int_t strip, Int_t cell);

  const std::string& GetName() const { return mGeoName; }

 private:
  static Geometry* sGeom; ///< Pointer to the unique instance of the singleton

  std::string mGeoName; ///< Geometry name string
};
} // namespace phos
} // namespace o2
#endif
