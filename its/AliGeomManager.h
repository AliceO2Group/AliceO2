#ifndef ALI_GEOM_MANAGER_H
#define ALI_GEOM_MANAGER_H

//
// Class for interfacing to the geometry; it also builds and manages the
// look-up tables for fast access to geometry and alignment information
// for sensitive alignable volumes:
// 1) the look-up table mapping unique volume ids to TGeoPNEntries
//    this allows to access directly by means of the unique index
//    the associated symbolic name and original global matrix
//    in addition to the functionality of the physical node
//    associated to a given alignable volume
// 2) the look-up table of the alignment objects associated to the
//    indexed alignable volumes
//

#include <TObject.h>

class TGeoManager;
class TGeoPNEntry;
class TGeoHMatrix;
class TObjArray;

class AliGeomManager: public TObject {

public:
  enum ELayerID{kInvalidLayer=0,
		kFirstLayer=1,
		kSPD1=1, kSPD2=2,
		kSDD1=3, kSDD2=4,
		kSSD1=5, kSSD2=6,
		kTPC1=7, kTPC2=8,
		kTRD1=9, kTRD2=10, kTRD3=11, kTRD4=12, kTRD5=13, kTRD6=14,
		kTOF=15,
		kPHOS1=16, kPHOS2=17,
		kHMPID=18,
		kMUON=19,
		kEMCAL=20,
		kLastLayer=21}; 
 
  static Bool_t GetOrigGlobalMatrix(const char *symname, TGeoHMatrix &m);

  ~AliGeomManager();

 private:
	AliGeomManager(); 
  static Bool_t       GetOrigGlobalMatrixFromPath(const char *path, TGeoHMatrix &m);

  static TGeoManager* fgGeometry;

  ClassDef(AliGeomManager, 0); // Manager of geometry information for alignment
};

#endif
