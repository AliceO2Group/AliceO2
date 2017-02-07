/// \file SimuClusterShaper.h
/// \brief Cluster shaper for the ALPIDE response simulation

#ifndef ALICEO2_ITS_SIMUCLUSTERSHAPER_H_
#define ALICEO2_ITS_SIMUCLUSTERSHAPER_H_

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to generate the cluster shape in the ITSU simulation    //
// Author: Davide Pagano                                         //
///////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <sstream>

#include "ITSSimulation/ClusterShape.h"

namespace AliceO2 {
  namespace ITS {

    class SimuClusterShaper : public TObject {

    public:
      SimuClusterShaper();
      SimuClusterShaper(const UInt_t &cs);
      virtual ~SimuClusterShaper();
      void FillClusterRandomly();
      void AddNoisePixel();

      inline UInt_t  GetNRows() {return fCShape->GetNRows();}
      inline UInt_t  GetNCols() {return fCShape->GetNCols();}
      inline void    GetShape(std::vector<UInt_t>& v) {fCShape->GetShape(v);}

      inline std::string ShapeSting(UInt_t cs, UInt_t *cshape) const {
        std::stringstream out;
        for (Int_t i = 0; i < cs; ++i) {
          out << cshape[i];
          if (i < cs-1) out << " ";
        }
        return out.str();
      }

    private:
      UInt_t fNpixOn;
      ClusterShape *fCShape;

      ClassDef(SimuClusterShaper,1)
    };
  }
}
#endif
