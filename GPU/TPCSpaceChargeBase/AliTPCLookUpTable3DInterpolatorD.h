#ifndef AliTPCLookUpTable3DInterpolatorD_H
#define AliTPCLookUpTable3DInterpolatorD_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


/// \class AliTPCLookUpTable3DInterpolatorD
/// \brief Wrap up look-up table for correction/distortion integral or derivative (electric field)
///        assume 3 components: r-component, phi-component and z-component
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Mar 4, 2015

#include "TMatrixD.h"
#include "AliTPC3DCylindricalInterpolator.h"

class AliTPCLookUpTable3DInterpolatorD {
public:

  AliTPCLookUpTable3DInterpolatorD();
  //AliTPCLookUpTable3DInterpolatorD(Int_t nRRow, Double_t rMin, Double_t rMax, Int_t nPhiSlice, Double_t phiMin, Double_t phiMax, Int_t nZColumn , Double_t zMin, Double_t zMax );
  AliTPCLookUpTable3DInterpolatorD(Int_t nRRow, TMatrixD**matricesRValue, Double_t *rList, Int_t nPhiSlice, TMatrixD**matricesPhiValue, Double_t *phiList, Int_t nZColumn, TMatrixD**matricesZValue, Double_t *zList, Int_t order);
  virtual ~AliTPCLookUpTable3DInterpolatorD();


	void SetNR(Int_t nRRow) {fNR = nRRow;}
	void SetNPhi(Int_t nPhiSlice) {fNPhi = nPhiSlice;}
	void SetNZ(Int_t nZColumn) {fNZ = nZColumn;}
	Int_t GetNR() {return fNR;}
	Int_t GetNPhi() {return fNPhi;}
	Int_t GetNZ() {return fNZ;}


	void SetRList(Double_t *rList) {fRList = rList; }
	void SetPhiList(Double_t *phiList) {fPhiList = phiList; }
	void SetZList(Double_t *zList) {fZList = zList; }
	void SetLookUpR(TMatrixD **matricesRValue) {fLookUpR = matricesRValue;}
	void SetLookUpPhi(TMatrixD **matricesPhiValue) {fLookUpPhi = matricesPhiValue;}
	void SetLookUpZ(TMatrixD **matricesZValue) {fLookUpZ = matricesZValue;}
	void SetOrder(Int_t order);
	void GetValue(Double_t r, Double_t phi, Double_t z, Double_t &rValue, Double_t &phiValue, Double_t &zValue);
  void GetValue(Double_t r, Double_t phi, Double_t z, Float_t &rValue, Float_t &phiValue, Float_t &zValue);
	void CopyFromMatricesToInterpolator();
	void CopyFromMatricesToInterpolator(Int_t iZ); // copy only iZ

	TMatrixD ** GetLookUpR() { return fLookUpR;}
	TMatrixD ** GetLookUpPhi() {return fLookUpPhi;}
	TMatrixD ** GetLookUpZ() {return fLookUpZ;}
	Double_t * GetRList() {return fRList;}
	Double_t * GetPhiList() {return fPhiList;}
	Double_t * GetZList() {return fZList;}



	AliTPC3DCylindricalInterpolator * GetInterpolatorR() {return fInterpolatorR;}
	AliTPC3DCylindricalInterpolator * GetInterpolatorPhi() {return fInterpolatorPhi;}
	AliTPC3DCylindricalInterpolator * GetInterpolatorZ() {return fInterpolatorZ;}
private:
	Int_t fOrder; ///< order of interpolation
	Int_t fNR; ///< number of grid in R
	Int_t fNPhi; ///< number of grid in Phi
	Int_t fNZ; ///< number of grid in Z
  
	TMatrixD **fLookUpR;   ///< Array to store distortion following the drift
  TMatrixD **fLookUpPhi;   ///< Array to store distortion following the drift
  TMatrixD **fLookUpZ;   ///< Array to store distortion following the drift
  
  AliTPC3DCylindricalInterpolator *fInterpolatorR; ///->Interpolator for R component
  AliTPC3DCylindricalInterpolator *fInterpolatorPhi; ///->Interpolator for Phi component
  AliTPC3DCylindricalInterpolator *fInterpolatorZ; ///-> Interpolator for Z component

	
	Double_t *fRList; //[fNR]List of R coordinate (regular grid)
	Double_t *fPhiList;  //[fNPhi]List of Phi coordinate (regular grid)
	Double_t *fZList; //[fNZ]List of Z coordinate (regular grid)
	
	Bool_t fIsAllocatingLookUp; ///< flag for initialization of cubic spline
	
  
	
/// \cond CLASSIMP
	ClassDef(AliTPCLookUpTable3DInterpolatorD,1);
/// \endcond
};


#endif
