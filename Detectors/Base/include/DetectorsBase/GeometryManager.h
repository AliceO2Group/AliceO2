// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryManager.h
/// \brief Definition of the GeometryManager class

#ifndef ALICEO2_BASE_GEOMETRYMANAGER_H_
#define ALICEO2_BASE_GEOMETRYMANAGER_H_

#include <TGeoPhysicalNode.h> // for TGeoPNEntry
#include <TObject.h>          // for TObject
#include "DetectorsBase/DetID.h"
#include "Rtypes.h" // for Bool_t, GeometryManager::Class, ClassDef, etc

class TGeoHMatrix; // lines 11-11
class TGeoManager; // lines 9-9

namespace o2
{
namespace Base
{
/// Class for interfacing to the geometry; it also builds and manages the look-up tables for fast
/// access to geometry and alignment information for sensitive alignable volumes:
/// 1) the look-up table mapping unique volume ids to TGeoPNEntries. This allows to access
/// directly by means of the unique index the associated symbolic name and original global matrix
/// in addition to the functionality of the physical node associated to a given alignable volume
/// 2) the look-up table of the alignment objects associated to the indexed alignable volumes
class GeometryManager : public TObject
{
 public:
  /// Get the global transformation matrix (ideal geometry) for a given alignable volume
  /// The alignable volume is identified by 'symname' which has to be either a valid symbolic
  /// name, the query being performed after alignment, or a valid volume path if the query is
  /// performed before alignment.
  static Bool_t getOriginalMatrix(DetID detid, int sensid, TGeoHMatrix& m);
  static Bool_t getOriginalMatrix(const char* symname, TGeoHMatrix& m);
  static const char* getSymbolicName(DetID detid, int sensid);
  static TGeoPNEntry* getPNEntry(DetID detid, Int_t sensid);
  static TGeoHMatrix* getMatrix(DetID detid, Int_t sensid);

  static int getSensID(DetID detid, int sensid)
  {
    /// compose combined detector+sensor ID for sensitive volumes
    return (detid.getMask() << sDetOffset) | (sensid & sSensorMask);
  }

  /// Default destructor
  ~GeometryManager() override = default;

 private:
  /// Default constructor
  GeometryManager();

  /// The method returns the global matrix for the volume identified by 'path' in the ideal
  /// detector geometry. The output global matrix is stored in 'm'.
  /// Returns kFALSE in case TGeo has not been initialized or the volume path is not valid.
  static Bool_t getOriginalMatrixFromPath(const char* path, TGeoHMatrix& m);

  static TGeoManager* sGeometry;

 protected:
  /// sensitive volume identifier composed from (det_mask<<sDetOffset)|(sensid&sSensorMask)
  static constexpr UInt_t sDetOffset = 16; /// detector identifier will start from this bit
  static constexpr UInt_t sSensorMask =
    (0x1 << sDetOffset) - 1; /// mask=max sensitive volumes allowed per detector (0xffff)

  ClassDefOverride(GeometryManager, 0); // Manager of geometry information for alignment
};
}
}

#endif
