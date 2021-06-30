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

#ifndef ALICEO2_PHOS_GEOMETRY_H_
#define ALICEO2_PHOS_GEOMETRY_H_

#include <string>
#include <array>

#include <Rtypes.h>
#include <RStringView.h>
#include <TGeoMatrix.h>
#include <TVector3.h>
#include <TMath.h>
#include <array>

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
  static Geometry* GetInstance() { return sGeom; }

  ///
  /// \return (newly created) pointer of the unique instance of the geometry. Previous instance is destroied.
  ///
  /// \param name: geometry name: PHOS (see main class description for definition)
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

  /// \breif Checks if two channels have common side
  /// \param absId1: absId of first channel, order important!
  /// \param absId2: absId of secont channel, order important!
  /// \return  0 are not neighbour but continue searching
  //         = 1 are neighbour
  //         = 2 are not neighbour but do not continue searching
  //         =-1 are not neighbour, continue searching, but do not look before d2 next time
  static int areNeighbours(short absId1, short absId2);

  /// \breif Converts Geant volume numbers to absId
  /// \return AbsId index of the PHOS cell
  /// \param moduleNumber: module number
  /// \param strip: strip number
  //  \param cell: cell in strip number
  static short relToAbsId(char moduleNumber, int strip, int cell);
  // Converts the absolute numbering into the following array
  //  relid[0] = PHOS Module number 1:module
  //  relid[1] = Row number inside a PHOS module (Phi coordinate)
  //  relid[2] = Column number inside a PHOS module (Z coordinate)
  static bool absToRelNumbering(short absId, char* relid);
  static char absIdToModule(short absId);
  static void absIdToRelPosInModule(short absId, float& x, float& z);
  static void relPosToRelId(short module, float x, float z, char* relId);
  static bool relToAbsNumbering(const char* RelId, short& AbsId);

  //Converters for TRU digits
  static bool truAbsToRelNumbering(short truId, char* relid);
  static short truRelToAbsNumbering(const char* relId);
  static bool truRelId2RelId(const char* truRelId, char* relId);
  static short relPosToTruId(char mod, float x, float z, short& ddl);

  //local position to absId
  static void relPosToAbsId(char module, float x, float z, short& absId);

  // convert local position in module to global position in ALICE
  void local2Global(char module, float x, float z, TVector3& globaPos) const;

  static int getTotalNCells() { return 56 * 64 * 4; } // TODO: evaluate from real geometry
  static bool isCellExists(short absId)
  {
    return absId >= 0 && absId <= getTotalNCells();
  } // TODO: evaluate from real geometry

  const std::string& GetName() const { return mGeoName; }

  const TGeoHMatrix* getAlignmentMatrix(int mod) const { return &(mPHOS[mod]); }

 private:
  static constexpr float CELLSTEP = 2.25;

  static Geometry* sGeom;           // Pointer to the unique instance of the singleton
  std::array<TGeoHMatrix, 5> mPHOS; //Rotation/shift matrices

  std::string mGeoName; ///< Geometry name string

  ClassDefNV(Geometry, 1);
};
} // namespace phos
} // namespace o2
#endif
