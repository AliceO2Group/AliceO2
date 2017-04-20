/// \file GeometryTGeo.h
/// \brief A simple interface class to TGeoManager
/// \author bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#ifndef ALICEO2_MFT_GEOMETRYTGEO_H_
#define ALICEO2_MFT_GEOMETRYTGEO_H_

#include "TObject.h"
#include "TString.h"

namespace o2 {
namespace MFT {

class GeometryTGeo : public TObject {

public:

  GeometryTGeo();
  ~GeometryTGeo() override;

  /// The number of disks
  Int_t getNDisks() const { return mNDisks; }
  /// The number of chips (sensors)
  Int_t getNChips() const {return mNChips;}  

  static const Char_t* getVolumeName()   { return sVolumeName.Data();   }
  static const Char_t* getHalfDetName()  { return sHalfDetName.Data();  }
  static const Char_t* getHalfDiskName() { return sHalfDiskName.Data(); }
  static const Char_t* getLadderName()   { return sLadderName.Data();   }
  static const Char_t* getSensorName()   { return sSensorName.Data();   }

private:

  GeometryTGeo(const GeometryTGeo &src);
  GeometryTGeo& operator=(const GeometryTGeo &);

  void build();

  Int_t  mNDisks;
  Int_t  mNChips;
  Int_t *mNLaddersHalfDisk;         //![2*mNDisks]

  static TString sVolumeName;      ///< \brief MFT-mother volume name
  static TString sHalfDetName;     ///< \brief MFT-half prefix
  static TString sHalfDiskName;    ///< \brief MFT-half disk prefix
  static TString sLadderName;      ///< \brief MFT-ladder prefix
  static TString sSensorName;      ///< \brief MFT-sensor (chip) prefix

  ClassDefOverride(GeometryTGeo, 1) // MFT geometry based on TGeo

};

}
}

#endif

