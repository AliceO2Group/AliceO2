/// \file GeometryTGeo.h
/// \brief A simple interface class to TGeoManager
/// \author bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#ifndef ALICEO2_MFT_GEOMETRYTGEO_H_
#define ALICEO2_MFT_GEOMETRYTGEO_H_

#include "TObject.h"
#include "TString.h"

namespace AliceO2 {
namespace MFT {

class GeometryTGeo : public TObject {

public:

  GeometryTGeo();
  ~GeometryTGeo();

  /// The number of disks
  Int_t GetNofDisks() const { return mNDisks; }
  /// The number of chips (sensors)
  Int_t  GetNChips() const {return mNChips;}  

  static const Char_t* GetVolumeName()   { return fgVolumeName.Data();   }
  static const Char_t* GetHalfDetName()  { return fgHalfDetName.Data();  }
  static const Char_t* GetHalfDiskName() { return fgHalfDiskName.Data(); }
  static const Char_t* GetLadderName()   { return fgLadderName.Data();   }
  static const Char_t* GetSensorName()   { return fgSensorName.Data();   }

private:

  GeometryTGeo(const GeometryTGeo &src);
  GeometryTGeo& operator=(const GeometryTGeo &);

  void Build();

  Int_t  mNDisks;
  Int_t  mNChips;
  Int_t *mNLaddersHalfDisk;         //![2*fNDisks]

  static TString fgVolumeName;      ///< \brief MFT-mother volume name
  static TString fgHalfDetName;     ///< \brief MFT-half prefix
  static TString fgHalfDiskName;    ///< \brief MFT-half disk prefix
  static TString fgLadderName;      ///< \brief MFT-ladder prefix
  static TString fgSensorName;      ///< \brief MFT-sensor (chip) prefix

  ClassDef(GeometryTGeo, 1) // MFT geometry based on TGeo

};

}
}

#endif

