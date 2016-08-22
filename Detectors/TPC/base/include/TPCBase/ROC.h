#ifndef AliceO2_TPC_ROC_H
#define AliceO2_TPC_ROC_H

#include "TPCBase/Defs.h"
#include "TPCBase/Sector.h"
//using namespace AliceO2::TPC;

namespace AliceO2 {
namespace TPC {
//   enum RocType {IROC=0, OROC=1};

  class ROC {
    public:
      enum { MaxROC=72 };
      ROC(){}
      ROC(unsigned char roc):mROC(roc%MaxROC){;}
      ROC(const ROC& roc)   :mROC(roc.mROC)  {;}

      ROC(const Sector& sec, const RocType type):mROC(sec.getSector() + (type==RocType::IROC)*SECTORSPERSIDE) {}
//       ROC(RocType t, Side s, unsigned char r):mROC( (s==Side::A)*SECTORSPERSIDE + (t==RocType::OROC)*SECTORSPERSIDE + r%SECTORSPERSIDE ) {}
//       ROC(Side t) {}

      ROC& operator= (const ROC& other) { mROC=other.mROC; return *this; }

      bool    operator==(const ROC& other) { return mROC==other.mROC; }
      bool    operator!=(const ROC& other) { return mROC!=other.mROC; }
      bool    operator< (const ROC& other)  { return mROC<other.mROC; }
      bool    operator++()                    { mLoop=++mROC>=MaxROC; mROC%=MaxROC; return mLoop; }

      unsigned char getRoc()  const { return mROC; }
      Side          side()    const { return (mROC/SECTORSPERSIDE)%SIDES?Side::C:Side::A; }
      RocType       rocType() const { return mROC<MaxROC/SIDES?RocType::IROC:RocType::OROC; }
      bool          looped()  const { return mLoop; }

    private:
      unsigned char mROC{};    /// ROC representation 0-MaxROC
      bool          mLoop{};   /// if operator execution resulted in looping
  };
}
}


#endif
