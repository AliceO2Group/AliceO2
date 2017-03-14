/// \file GeometryBuilder.h
/// \brief Class describing MFT Geometry Builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_GEOMETRYBUILDER_H_
#define ALICEO2_MFT_GEOMETRYBUILDER_H_

#include "TNamed.h"

namespace AliceO2 {
namespace MFT {

class GeometryBuilder : public TNamed {

 public:

  GeometryBuilder();
  virtual ~GeometryBuilder();

  void BuildGeometry();

 private:

  ClassDef(GeometryBuilder, 1)

};

}
}

#endif
