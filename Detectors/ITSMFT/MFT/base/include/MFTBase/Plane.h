/// \file Plane.h
/// \brief Class for the description of the structure for the planes of the ALICE Muon Forward Tracker
/// \author Antonio Uras <antonio.uras@cern.ch>

#ifndef ALICEO2_MFT_PLANE_H_
#define ALICEO2_MFT_PLANE_H_

#include "TNamed.h"
#include "THnSparse.h"
#include "TAxis.h"

#include "FairLogger.h"

class TClonesArray;

namespace o2 {
namespace MFT {

class Plane : public TNamed {

public:

  Plane();
  Plane(const Char_t *name, const Char_t *title);
  Plane(const Plane& pt);
  Plane& operator=(const Plane &source);
  
  ~Plane() override;  // destructor
  void Clear(const Option_t* /*opt*/) override;
  
  Bool_t init(Int_t    planeNumber,
	      Double_t zCenter, 
	      Double_t rMin, 
	      Double_t rMax, 
	      Double_t pixelSizeX, 
	      Double_t pixelSizeY, 
	      Double_t thicknessActive, 
	      Double_t thicknessSupport, 
	      Double_t thicknessReadout,
	      Bool_t   hasPixelRectangularPatternAlongY);
  
  Bool_t createStructure();

  Int_t getNActiveElements()  const { return mActiveElements->GetEntries();  }
  Int_t getNReadoutElements() const { return mReadoutElements->GetEntries(); }
  Int_t getNSupportElements() const { return mSupportElements->GetEntries(); }

  TClonesArray* getActiveElements()  { return mActiveElements;  }
  TClonesArray* getReadoutElements() { return mReadoutElements; }
  TClonesArray* getSupportElements() { return mSupportElements; }

  THnSparseC* getActiveElement(Int_t id);
  THnSparseC* getReadoutElement(Int_t id);
  THnSparseC* getSupportElement(Int_t id);

  Bool_t isFront(THnSparseC *element) const { return (element->GetAxis(2)->GetXmin() < mZCenter); }

  void drawPlane(Option_t *opt="");

  Double_t getRMinSupport() const { return mRMinSupport; }
  Double_t getRMaxSupport() const { return mRMaxSupport; }
  Double_t getThicknessSupport() { return getSupportElement(0)->GetAxis(2)->GetXmax() - getSupportElement(0)->GetAxis(2)->GetXmin(); }
  
  Double_t getZCenter()            const { return mZCenter; }
  Double_t getZCenterActiveFront() const { return mZCenterActiveFront; }
  Double_t getZCenterActiveBack()  const { return mZCenterActiveBack; }

  void setEquivalentSilicon(Double_t equivalentSilicon)                       { mEquivalentSilicon            = equivalentSilicon; }
  void setEquivalentSiliconBeforeFront(Double_t equivalentSiliconBeforeFront) { mEquivalentSiliconBeforeFront = equivalentSiliconBeforeFront; }
  void setEquivalentSiliconBeforeBack(Double_t equivalentSiliconBeforeBack)   { mEquivalentSiliconBeforeBack  = equivalentSiliconBeforeBack; }
  Double_t getEquivalentSilicon()            const { return mEquivalentSilicon; }
  Double_t getEquivalentSiliconBeforeFront() const { return mEquivalentSiliconBeforeFront; }
  Double_t getEquivalentSiliconBeforeBack()  const { return mEquivalentSiliconBeforeBack; }

  Int_t getNumberOfChips(Option_t *opt);
  Bool_t hasPixelRectangularPatternAlongY() { return mHasPixelRectangularPatternAlongY; }
  
private:

  Int_t mPlaneNumber;

  Double_t mZCenter, mRMinSupport, mRMax, mRMaxSupport, mPixelSizeX, mPixelSizeY, mThicknessActive, mThicknessSupport, mThicknessReadout;
  Double_t mZCenterActiveFront, mZCenterActiveBack, mEquivalentSilicon, mEquivalentSiliconBeforeFront, mEquivalentSiliconBeforeBack;

  TClonesArray *mActiveElements, *mReadoutElements, *mSupportElements;

  Bool_t mHasPixelRectangularPatternAlongY, mPlaneIsOdd;

  ClassDefOverride(Plane, 1)

};

}
}

#endif

