/// \file ClusterShape.h
/// \brief Cluster shape class for the ALPIDE response simulation

#ifndef ALICEO2_ITS_CLUSTERSHAPE_H_
#define ALICEO2_ITS_CLUSTERSHAPE_H_

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to describe the cluster shape in the ITSU simulation    //
// Author: Davide Pagano                                         //
///////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <sstream>

namespace AliceO2 {
namespace ITS {

class ClusterShape : public TObject {

 public:
  ClusterShape();
  ClusterShape(UInt_t, UInt_t, UInt_t);
  virtual ~ClusterShape();

  inline void SetNRows(UInt_t Nrows) {fNrows = Nrows;}
  inline void SetNCols(UInt_t Ncols) {fNcols = Ncols;}
  inline void SetNFiredPixels(UInt_t NFPix) {
    fNFPix = NFPix;
    fShape = new UInt_t[fNFPix];
  }
  inline void SetShapeValue(UInt_t index, UInt_t value) {
    fShape[index] = value;
  }

  // returns an unique ID based on the cluster size and shape
  Long64_t GetShapeID();

  inline UInt_t GetNRows() const {return fNrows;}
  inline UInt_t GetNCols() const {return fNcols;}
  inline UInt_t GetNFiredPixels() const {return fNFPix;}
  inline UInt_t GetShapeValue(UInt_t index) {return fShape[index];}
  inline UInt_t* GetShape() const {return fShape;}

  inline Bool_t HasElement(UInt_t value) {
    for (UInt_t i = 0; i < fNFPix; ++i) {
      if (fShape[i] > value) break;
      if (fShape[i] == value) return true;
    }
    return false;
  }

  inline std::string ShapeSting() const {
    return ShapeSting(fNFPix, fShape);
  }

  static std::string ShapeSting(UInt_t nFPix, UInt_t *shape) {
    std::stringstream out;
    for (UInt_t i = 0; i < nFPix; ++i) {
      out << shape[i];
      if (i < nFPix-1) out << " ";
    }
    return out.str();
  }

 private:
  UInt_t  fNrows;
  UInt_t  fNcols;
  UInt_t  fNFPix;
  UInt_t *fShape;

  ClassDef(ClusterShape,1)
};
}
}
#endif
