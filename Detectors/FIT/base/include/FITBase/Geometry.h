// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FIT_GEOMETRY_H_
#define ALICEO2_FIT_GEOMETRY_H_
////////////////////////////////////////////////
// Full geomrtry  hits classes for detector: FIT    //
////////////////////////////////////////////////

#include <Rtypes.h>
#include <TNamed.h>
namespace o2
{
namespace FIT
{

class Geometry : public TNamed
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
  ///
  Geometry(const Text_t* name, const Text_t* title = "", const Text_t* mcname = "", const Text_t* mctitle = "");

  ///
  /// Copy constructor.
  ///
  Geometry(const Geometry& geom);

  ///
  /// Destructor.
  ///
  ~Geometry() ;

  ///
  /// Assign operator.
  ///
  Geometry& operator=(const Geometry& rvalue);

  ///
  /// \return the pointer of the unique instance of the geometry
  ///
  /// It should have been set before.
  ///
  static Geometry* GetInstance();

  ///
  /// \return the pointer of the unique instance of the geometry
  ///
    ///
  //  static Geometry* GetInstance(const Text_t* name, const Text_t* title = "", const Text_t* mcname = "TGeant3",
  //                               const Text_t* mctitle = "");

  ///
  /// Instanciate geometry depending on the run number. Mostly used in analysis and MC anchors.
  ///
  /// \return the pointer of the unique instance
  ///
//////////
  // General
  //
  //  static Bool_t IsInitialized() { return Geometry::sGeom != nullptr; }
  // static const Char_t* GetDefaultGeometryName() {return EMCGeometry::fgkDefaultGeometryName;}

 private:
  static Geometry* sGeom;                    ///< Pointer to the unique instance of the singleton
  static const Char_t* sDefaultGeometryName; ///< Default name of geometry
  
 private: 
  ClassDef(Geometry, 1);
};
}
}
#endif
