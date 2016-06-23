#ifndef AliceO2_TPC_ROC_H
#define AliceO2_TPC_ROC_H

#include "Defs.h"
//using namespace AliceO2::TPC;

namespace AliceO2 {
namespace TPC {
//   enum RocType {IROC=0, OROC=1};

  class ROC {
    public:
      enum { MaxROC=72 };
      ROC(){}
      ROC(unsigned char sec):mROC(sec%MaxROC){;}
      ROC(const ROC& sec):mROC(sec.mROC){;}
      ROC(RocType t, Side s, unsigned char r):mROC( (s==Side::A)*18 + (t==RocType::OROC)*18 + r%18 ) {}
//       ROC(Side t) {}

      ROC& operator= (const ROC& other) { mROC=other.mROC; return *this; }

      bool    operator==(const ROC& other) { return mROC==other.mROC; }
      bool    operator!=(const ROC& other) { return mROC!=other.mROC; }
      bool    operator< (const ROC& other)  { return mROC<other.mROC; }
      bool    operator++()                    { mLoop=++mROC>=MaxROC; mROC%=MaxROC; return mLoop; }

      unsigned char roc() const { return mROC; }
      Side    side()      const { return (mROC/18)%2?Side::C:Side::A; }
      RocType rocType()   const { return mROC<MaxROC/2?RocType::IROC:RocType::OROC; }
      bool    looped()    const { return mLoop; }

    private:
      unsigned char mROC{};    /// ROC representation 0-MaxROC
      bool          mLoop{};   /// if operator execution resulted in looping
  };
}
}


#endif
