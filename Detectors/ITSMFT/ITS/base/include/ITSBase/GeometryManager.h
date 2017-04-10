/// \file GeometryManager.h
/// \brief Definition of the GeometryManager class

#ifndef ALICEO2_ITS_GEOMETRYMANAGER_H_
#define ALICEO2_ITS_GEOMETRYMANAGER_H_

#include <TObject.h> // for TObject
#include "Rtypes.h"  // for Bool_t, GeometryManager::Class, ClassDef, etc

class TGeoHMatrix; // lines 11-11
class TGeoManager; // lines 9-9

namespace o2
{
namespace ITS
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
  enum ELayerID {
    kInvalidLayer = 0,
    kFirstLayer = 1,
    kSPD1 = 1,
    kSPD2 = 2,
    kSDD1 = 3,
    kSDD2 = 4,
    kSSD1 = 5,
    kSSD2 = 6,
    kTPC1 = 7,
    kTPC2 = 8,
    kTRD1 = 9,
    kTRD2 = 10,
    kTRD3 = 11,
    kTRD4 = 12,
    kTRD5 = 13,
    kTRD6 = 14,
    kTOF = 15,
    kPHOS1 = 16,
    kPHOS2 = 17,
    kHMPID = 18,
    kMUON = 19,
    kEMCAL = 20,
    kLastLayer = 21
  };

  /// Get the global transformation matrix (ideal geometry) for a given alignable volume
  /// The alignable volume is identified by 'symname' which has to be either a valid symbolic
  /// name, the query being performed after alignment, or a valid volume path if the query is
  /// performed before alignment.
  static Bool_t getOriginalGlobalMatrix(const char* symname, TGeoHMatrix& m);

  /// Default destructor
  ~GeometryManager() override;

 private:
  /// Default constructor
  GeometryManager();

  /// The method returns the global matrix for the volume identified by 'path' in the ideal
  /// detector geometry. The output global matrix is stored in 'm'.
  /// Returns kFALSE in case TGeo has not been initialized or the volume path is not valid.
  static Bool_t getOriginalGlobalMatrixFromPath(const char* path, TGeoHMatrix& m);

  static TGeoManager* mGeometry;

  ClassDefOverride(GeometryManager, 0); // Manager of geometry information for alignment
};
}
}

#endif
