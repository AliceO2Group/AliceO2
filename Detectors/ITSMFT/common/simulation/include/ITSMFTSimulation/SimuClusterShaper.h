/// \file SimuClusterShaper.h
/// \brief Cluster shaper for the ALPIDE response simulation

#ifndef ALICEO2_ITSMFT_SIMUCLUSTERSHAPER_H_
#define ALICEO2_ITSMFT_SIMUCLUSTERSHAPER_H_

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to generate the cluster shape in the ITSU simulation    //
// Author: Davide Pagano                                         //
///////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <sstream>

#include "ITSMFTSimulation/ClusterShape.h"

namespace o2 {
  namespace ITSMFT {

    class SimuClusterShaper : public TObject {

    public:
      SimuClusterShaper();
      SimuClusterShaper(const UInt_t &cs);
      ~SimuClusterShaper() override;
      void FillClusterRandomly();
      void AddNoisePixel();

      inline UInt_t  GetNRows() {return mCShape->GetNRows();}
      inline UInt_t  GetNCols() {return mCShape->GetNCols();}
      inline void    GetShape(std::vector<UInt_t>& v) {mCShape->GetShape(v);}

      inline std::string ShapeSting(UInt_t cs, UInt_t *cshape) const {
        std::stringstream out;
        for (Int_t i = 0; i < cs; ++i) {
          out << cshape[i];
          if (i < cs-1) out << " ";
        }
        return out.str();
      }

    private:
      UInt_t mNpixOn;
      ClusterShape *mCShape;

      ClassDefOverride(SimuClusterShaper,1)
    };
  }
}
#endif
