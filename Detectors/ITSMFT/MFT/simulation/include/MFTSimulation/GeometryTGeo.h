/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class
/// \author bogdan.vulpescu@cern.ch - 01/08/2016

#ifndef ALICEO2_MFT_GEOMETRYTGEO_H_
#define ALICEO2_MFT_GEOMETRYTGEO_H_

#include "TObject.h"

namespace AliceO2 {
namespace MFT {

class GeometryTGeo : public TObject {

public:

  GeometryTGeo();

  virtual ~GeometryTGeo();

  GeometryTGeo(const GeometryTGeo& src);

  GeometryTGeo& operator=(const GeometryTGeo& geom);

  void Build();

  /// Returns the number of layers
  Int_t getNofDisks() const
  {
    return mNofDisks;
  }

private:

  Int_t mNofDisks;

  ClassDef(GeometryTGeo, 1) // MFT geometry based on TGeo

};

}
}

#endif

