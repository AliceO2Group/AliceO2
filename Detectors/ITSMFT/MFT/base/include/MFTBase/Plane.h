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
  
  virtual ~Plane();  // destructor
  virtual void Clear(const Option_t* /*opt*/);
  
  Bool_t Init(Int_t    planeNumber,
	      Double_t zCenter, 
	      Double_t rMin, 
	      Double_t rMax, 
	      Double_t pixelSizeX, 
	      Double_t pixelSizeY, 
	      Double_t thicknessActive, 
	      Double_t thicknessSupport, 
	      Double_t thicknessReadout,
	      Bool_t   hasPixelRectangularPatternAlongY);
  
  Bool_t CreateStructure();

  Int_t GetNActiveElements()  const { return mActiveElements->GetEntries();  }
  Int_t GetNReadoutElements() const { return mReadoutElements->GetEntries(); }
  Int_t GetNSupportElements() const { return mSupportElements->GetEntries(); }

  TClonesArray* GetActiveElements()  { return mActiveElements;  }
  TClonesArray* GetReadoutElements() { return mReadoutElements; }
  TClonesArray* GetSupportElements() { return mSupportElements; }

  THnSparseC* GetActiveElement(Int_t id);
  THnSparseC* GetReadoutElement(Int_t id);
  THnSparseC* GetSupportElement(Int_t id);

  Bool_t IsFront(THnSparseC *element) const { return (element->GetAxis(2)->GetXmin() < mZCenter); }

  void DrawPlane(Option_t *opt="");

  Double_t GetRMinSupport() const { return mRMinSupport; }
  Double_t GetRMaxSupport() const { return mRMaxSupport; }
  Double_t GetThicknessSupport() { return GetSupportElement(0)->GetAxis(2)->GetXmax() - GetSupportElement(0)->GetAxis(2)->GetXmin(); }
  
  Double_t GetZCenter()            const { return mZCenter; }
  Double_t GetZCenterActiveFront() const { return mZCenterActiveFront; }
  Double_t GetZCenterActiveBack()  const { return mZCenterActiveBack; }

  void SetEquivalentSilicon(Double_t equivalentSilicon)                       { mEquivalentSilicon            = equivalentSilicon; }
  void SetEquivalentSiliconBeforeFront(Double_t equivalentSiliconBeforeFront) { mEquivalentSiliconBeforeFront = equivalentSiliconBeforeFront; }
  void SetEquivalentSiliconBeforeBack(Double_t equivalentSiliconBeforeBack)   { mEquivalentSiliconBeforeBack  = equivalentSiliconBeforeBack; }
  Double_t GetEquivalentSilicon()            const { return mEquivalentSilicon; }
  Double_t GetEquivalentSiliconBeforeFront() const { return mEquivalentSiliconBeforeFront; }
  Double_t GetEquivalentSiliconBeforeBack()  const { return mEquivalentSiliconBeforeBack; }

  Int_t GetNumberOfChips(Option_t *opt);
  Bool_t HasPixelRectangularPatternAlongY() { return mHasPixelRectangularPatternAlongY; }
  
private:

  Int_t mPlaneNumber;

  Double_t mZCenter, mRMinSupport, mRMax, mRMaxSupport, mPixelSizeX, mPixelSizeY, mThicknessActive, mThicknessSupport, mThicknessReadout;
  Double_t mZCenterActiveFront, mZCenterActiveBack, mEquivalentSilicon, mEquivalentSiliconBeforeFront, mEquivalentSiliconBeforeBack;

  TClonesArray *mActiveElements, *mReadoutElements, *mSupportElements;

  Bool_t mHasPixelRectangularPatternAlongY, mPlaneIsOdd;

  ClassDef(Plane, 1)

};

}
}

#endif

