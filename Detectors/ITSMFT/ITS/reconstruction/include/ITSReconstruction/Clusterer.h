/// \file Clusterer.h
/// \brief Task for ALICE ITS clusterisation
#ifndef ALICEO2_ITS_Clusterer_H_
#define ALICEO2_ITS_Clusterer_H_

#include "Rtypes.h"   // for Clusterer::Class, Double_t, ClassDef, etc
#include "TObject.h"  // for TObject
#include "ITSBase/GeometryTGeo.h"

class TClonesArray;

namespace AliceO2 {

namespace ITS {

class Clusterer : public TObject
{
  public:
    Clusterer();
   ~Clusterer();

    void Init(Bool_t build=kTRUE);

    /// Steer conversion of points to digits
    /// @param points Container with ITS points
    /// @return digits container
   void Process(const TClonesArray *digits, TClonesArray *clusters);

  private:
    Clusterer(const Clusterer &);
    Clusterer &operator=(const Clusterer &);

    GeometryTGeo fGeometry;            ///< ITS upgrade geometry

  ClassDef(Clusterer, 2);
};
}
}

#endif /* ALICEO2_ITS_Clusterer_H_ */
