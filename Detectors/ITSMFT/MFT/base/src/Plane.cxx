/// \file Plane.cxx
/// \brief Class for the description of the structure for the planes of the ALICE Muon Forward Tracker
/// \author Antonio Uras <antonio.uras@cern.ch>

#include "TNamed.h"
#include "THnSparse.h"
#include "TClonesArray.h"
#include "TAxis.h"
#include "TPave.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TEllipse.h"
#include "TMath.h"

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Plane.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Plane)
/// \endcond

//_____________________________________________________________________________
Plane::Plane():
  TNamed(),
  mPlaneNumber(-1),
  mZCenter(0), 
  mRMinSupport(0), 
  mRMax(0),
  mRMaxSupport(0),
  mPixelSizeX(0), 
  mPixelSizeY(0), 
  mThicknessActive(0), 
  mThicknessSupport(0), 
  mThicknessReadout(0),
  mZCenterActiveFront(0),
  mZCenterActiveBack(0),
  mEquivalentSilicon(0),
  mEquivalentSiliconBeforeFront(0),
  mEquivalentSiliconBeforeBack(0),
  mActiveElements(nullptr),
  mReadoutElements(nullptr),
  mSupportElements(nullptr),
  mHasPixelRectangularPatternAlongY(kFALSE),
  mPlaneIsOdd(kFALSE)
{

  // default constructor

}

//_____________________________________________________________________________
Plane::Plane(const Char_t *name, const Char_t *title):
  TNamed(name, title),
  mPlaneNumber(-1),
  mZCenter(0), 
  mRMinSupport(0), 
  mRMax(0),
  mRMaxSupport(0),
  mPixelSizeX(0), 
  mPixelSizeY(0), 
  mThicknessActive(0), 
  mThicknessSupport(0), 
  mThicknessReadout(0),
  mZCenterActiveFront(0),
  mZCenterActiveBack(0),
  mEquivalentSilicon(0),
  mEquivalentSiliconBeforeFront(0),
  mEquivalentSiliconBeforeBack(0),
  mActiveElements(nullptr),
  mReadoutElements(nullptr),
  mSupportElements(nullptr),
  mHasPixelRectangularPatternAlongY(kFALSE),
  mPlaneIsOdd(kFALSE)
{

  // constructor
  mActiveElements  = new TClonesArray("THnSparseC");
  mReadoutElements = new TClonesArray("THnSparseC");
  mSupportElements = new TClonesArray("THnSparseC");
  mActiveElements->SetOwner(kTRUE);
  mReadoutElements->SetOwner(kTRUE);
  mSupportElements->SetOwner(kTRUE);
  
}

//_____________________________________________________________________________
Plane::Plane(const Plane& plane):
  TNamed(plane),
  mPlaneNumber(plane.mPlaneNumber),
  mZCenter(plane.mZCenter), 
  mRMinSupport(plane.mRMinSupport), 
  mRMax(plane.mRMax),
  mRMaxSupport(plane.mRMaxSupport),
  mPixelSizeX(plane.mPixelSizeX), 
  mPixelSizeY(plane.mPixelSizeY), 
  mThicknessActive(plane.mThicknessActive), 
  mThicknessSupport(plane.mThicknessSupport), 
  mThicknessReadout(plane.mThicknessReadout),
  mZCenterActiveFront(plane.mZCenterActiveFront),
  mZCenterActiveBack(plane.mZCenterActiveBack),
  mEquivalentSilicon(plane.mEquivalentSilicon),
  mEquivalentSiliconBeforeFront(plane.mEquivalentSiliconBeforeFront),
  mEquivalentSiliconBeforeBack(plane.mEquivalentSiliconBeforeBack),
  mActiveElements(nullptr),
  mReadoutElements(nullptr),
  mSupportElements(nullptr),
  mHasPixelRectangularPatternAlongY(plane.mHasPixelRectangularPatternAlongY),
  mPlaneIsOdd(plane.mPlaneIsOdd)
{

  // copy constructor
  mActiveElements  = new TClonesArray(*(plane.mActiveElements));
  mActiveElements  -> SetOwner(kTRUE);
  mReadoutElements = new TClonesArray(*(plane.mReadoutElements));
  mReadoutElements -> SetOwner(kTRUE);
  mSupportElements = new TClonesArray(*(plane.mSupportElements));
  mSupportElements -> SetOwner(kTRUE);

	
}

//_____________________________________________________________________________
Plane::~Plane() 
{

  Info("~Plane","Delete Plane",0,0);
  if(mActiveElements) mActiveElements->Delete();
  delete mActiveElements; 
  if(mReadoutElements) mReadoutElements->Delete();
  delete mReadoutElements; 
  if(mSupportElements) mSupportElements->Delete();
  delete mSupportElements; 

}

//_____________________________________________________________________________
void Plane::Clear(const Option_t* /*opt*/) 
{

  Info("Clear","Clear Plane",0,0);
  if(mActiveElements) mActiveElements->Delete();
  delete mActiveElements; mActiveElements=nullptr;
  if(mReadoutElements) mReadoutElements->Delete();
  delete mReadoutElements;  mReadoutElements=nullptr; 
  if(mSupportElements) mSupportElements->Delete();
  delete mSupportElements;   mSupportElements=nullptr;

}

//_____________________________________________________________________________
Plane& Plane::operator=(const Plane& plane) 
{

  // Assignment operator
  
  // check assignement to self
  if (this != &plane) {
    
    // base class assignement
    TNamed::operator=(plane);
    
    // clear memory
    Clear("");
    
    mPlaneNumber                      = plane.mPlaneNumber;
    mZCenter                          = plane.mZCenter; 
    mRMinSupport                      = plane.mRMinSupport; 
    mRMax                             = plane.mRMax;
    mRMaxSupport                      = plane.mRMaxSupport;
    mPixelSizeX                       = plane.mPixelSizeX;
    mPixelSizeY                       = plane.mPixelSizeY; 
    mThicknessActive                  = plane.mThicknessActive; 
    mThicknessSupport                 = plane.mThicknessSupport; 
    mThicknessReadout                 = plane.mThicknessReadout;
    mZCenterActiveFront               = plane.mZCenterActiveFront;
    mZCenterActiveBack                = plane.mZCenterActiveBack;
    mEquivalentSilicon                = plane.mEquivalentSilicon;
    mEquivalentSiliconBeforeFront     = plane.mEquivalentSiliconBeforeFront;
    mEquivalentSiliconBeforeBack      = plane.mEquivalentSiliconBeforeBack;
    mActiveElements = new TClonesArray(*(plane.mActiveElements));
    mActiveElements -> SetOwner(kTRUE);
    mReadoutElements = new TClonesArray(*(plane.mReadoutElements));
    mReadoutElements -> SetOwner(kTRUE);
    mSupportElements = new TClonesArray(*(plane.mSupportElements));
    mSupportElements -> SetOwner(kTRUE);
    mHasPixelRectangularPatternAlongY = plane.mHasPixelRectangularPatternAlongY;
    mPlaneIsOdd                       = plane.mPlaneIsOdd;

  }
  
  return *this;
  
}

//_____________________________________________________________________________
Bool_t Plane::Init(Int_t    planeNumber,
			 Double_t zCenter, 
			 Double_t rMin, 
			 Double_t rMax, 
			 Double_t pixelSizeX, 
			 Double_t pixelSizeY, 
			 Double_t thicknessActive, 
			 Double_t thicknessSupport, 
			 Double_t thicknessReadout,
			 Bool_t   hasPixelRectangularPatternAlongY) 
{

  LOG(DEBUG1) << "Init: " << Form("initializing plane structure for plane %s", GetName()) << FairLogger::endl;

  mPlaneNumber      = planeNumber;
  mZCenter          = zCenter;
  mRMinSupport      = rMin;
  mRMax             = rMax;
  mPixelSizeX       = pixelSizeX;
  mPixelSizeY       = pixelSizeY;
  mThicknessActive  = thicknessActive;
  mThicknessSupport = thicknessSupport;
  mThicknessReadout = thicknessReadout;

  mHasPixelRectangularPatternAlongY = hasPixelRectangularPatternAlongY;

  mZCenterActiveFront = mZCenter - 0.5*mThicknessSupport - 0.5*mThicknessActive;
  mZCenterActiveBack  = mZCenter + 0.5*mThicknessSupport + 0.5*mThicknessActive;

  if (mRMax < mRMinSupport+Constants::fHeightActive) mRMax = mRMinSupport + Constants::fHeightActive;

  Int_t nLaddersWithinPipe = Int_t(mRMinSupport/(Constants::fHeightActive-Constants::fActiveSuperposition));
  if (mRMinSupport-nLaddersWithinPipe*(Constants::fHeightActive-Constants::fActiveSuperposition) > 0.5*(Constants::fHeightActive-2*Constants::fActiveSuperposition)) mPlaneIsOdd = kTRUE;
  else mPlaneIsOdd = kFALSE;

  mRMax = mRMinSupport + (Constants::fHeightActive-Constants::fActiveSuperposition) * 
    (Int_t((mRMax-mRMinSupport-Constants::fHeightActive)/(Constants::fHeightActive-Constants::fActiveSuperposition))+1) + Constants::fHeightActive;

  mRMaxSupport = TMath::Sqrt(Constants::fHeightActive*(2.*rMax-Constants::fHeightActive) + mRMax*mRMax) + Constants::fSupportExtMargin;
   
  return kTRUE;
 
}

//_____________________________________________________________________________
Bool_t Plane::CreateStructure() 
{

  Int_t nBins[3]={0};
  Double_t minPosition[3]={0}, maxPosition[3]={0};
  
  // ------------------- det elements: active + readout ----------------------------------

  Double_t lowEdgeActive = -1.*mRMax;
  Double_t supEdgeActive = lowEdgeActive + Constants::fHeightActive;
  Double_t zMinFront = mZCenter - 0.5*mThicknessSupport - mThicknessActive;
  Double_t zMinBack  = mZCenter + 0.5*mThicknessSupport;
  Double_t zMin = 0.;
  Bool_t isFront = kTRUE;
  
  while (lowEdgeActive < 0) {
    
    Double_t extLimitAtLowEdgeActive = TMath::Sqrt((mRMax-TMath::Abs(lowEdgeActive)) * TMath::Abs(2*mRMax - (mRMax-TMath::Abs(lowEdgeActive))));
    Double_t extLimitAtSupEdgeActive = TMath::Sqrt((mRMax-TMath::Abs(supEdgeActive)) * TMath::Abs(2*mRMax - (mRMax-TMath::Abs(supEdgeActive))));

    // creating new det element: active + readout
    
    Double_t extLimitDetElem = TMath::Max(extLimitAtLowEdgeActive, extLimitAtSupEdgeActive);
    
    if (supEdgeActive<-1.*mRMinSupport+0.01 || lowEdgeActive>1.*mRMinSupport-0.01) {     // single element covering the row
      
      nBins[0] = TMath::Nint(2.*extLimitDetElem/mPixelSizeX);
      nBins[1] = TMath::Nint(Constants::fHeightActive/mPixelSizeY);
      nBins[2] = 1;

      // element below the pipe
      
      if (isFront) zMin = zMinFront;
      else         zMin = zMinBack;

      minPosition[0] = -1.*extLimitDetElem;
      minPosition[1] = lowEdgeActive;
      minPosition[2] = zMin;
      
      maxPosition[0] = +1.*extLimitDetElem;
      maxPosition[1] = supEdgeActive;
      maxPosition[2] = zMin+mThicknessActive; 
      
      new ((*mActiveElements)[mActiveElements->GetEntries()]) THnSparseC(Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 3, nBins, minPosition, maxPosition);

      minPosition[1] = lowEdgeActive-Constants::fHeightReadout;
      maxPosition[1] = lowEdgeActive;
      
      new ((*mReadoutElements)[mReadoutElements->GetEntries()]) THnSparseC(Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   3, nBins, minPosition, maxPosition);

      // specular element above the pipe

      if (mPlaneIsOdd) {
	if (isFront) zMin = zMinBack;
	else         zMin = zMinFront;
      }

      minPosition[0] = -1.*extLimitDetElem;
      minPosition[1] = -1.*supEdgeActive;
      minPosition[2] = zMin;
      
      maxPosition[0] = +1.*extLimitDetElem;
      maxPosition[1] = -1.*lowEdgeActive;
      maxPosition[2] = zMin+mThicknessActive; 
      
      new ((*mActiveElements)[mActiveElements->GetEntries()]) THnSparseC(Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 3, nBins, minPosition, maxPosition);

      minPosition[1] = -1.*lowEdgeActive;
      maxPosition[1] = -1.*(lowEdgeActive-Constants::fHeightReadout);

      new ((*mReadoutElements)[mReadoutElements->GetEntries()]) THnSparseC(Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   3, nBins, minPosition, maxPosition);

    }
    
    else {     // two elements covering the row
      
      Double_t intLimitAtLowEdge = 0., intLimitAtSupEdge = 0.;
      if (mRMinSupport-TMath::Abs(lowEdgeActive)>0.) intLimitAtLowEdge = TMath::Sqrt((mRMinSupport-TMath::Abs(lowEdgeActive)) * TMath::Abs(2*mRMinSupport - (mRMinSupport-TMath::Abs(lowEdgeActive))));
      if (mRMinSupport-TMath::Abs(supEdgeActive)>0.) intLimitAtSupEdge = TMath::Sqrt((mRMinSupport-TMath::Abs(supEdgeActive)) * TMath::Abs(2*mRMinSupport - (mRMinSupport-TMath::Abs(supEdgeActive))));
      Double_t intLimitDetElem = TMath::Max(intLimitAtLowEdge, intLimitAtSupEdge);
      
      nBins[0] = TMath::Nint((extLimitDetElem-intLimitDetElem)/mPixelSizeX);
      nBins[1] = TMath::Nint(Constants::fHeightActive/mPixelSizeY);
      nBins[2] = 1;
      
      // left element: y < 0
      
      if (isFront) zMin = zMinFront;
      else         zMin = zMinBack;

      minPosition[0] = -1.*extLimitDetElem;
      minPosition[1] = lowEdgeActive;
      minPosition[2] = zMin;
      
      maxPosition[0] = -1.*intLimitDetElem;
      maxPosition[1] = supEdgeActive;
      maxPosition[2] = zMin+mThicknessActive; 
      
      new ((*mActiveElements)[mActiveElements->GetEntries()]) THnSparseC(Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 3, nBins, minPosition, maxPosition);	
      
      minPosition[1] = lowEdgeActive-Constants::fHeightReadout;
      maxPosition[1] = lowEdgeActive;
      
      new ((*mReadoutElements)[mReadoutElements->GetEntries()]) THnSparseC(Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   3, nBins, minPosition, maxPosition);

      // left element: y > 0
      
      if (supEdgeActive < 0.5*Constants::fHeightActive) {
	
	if (mPlaneIsOdd) {
	  if (isFront) zMin = zMinBack;
	  else         zMin = zMinFront;
	}
	
	minPosition[0] = -1.*extLimitDetElem;
	minPosition[1] = -1.*supEdgeActive;
	minPosition[2] = zMin;
	
	maxPosition[0] = -1.*intLimitDetElem;
	maxPosition[1] = -1.*lowEdgeActive;
	maxPosition[2] = zMin+mThicknessActive; 
	
	new ((*mActiveElements)[mActiveElements->GetEntries()]) THnSparseC(Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									   Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									   3, nBins, minPosition, maxPosition);	
	
	minPosition[1] = -1.*lowEdgeActive;
	maxPosition[1] = -1.*(lowEdgeActive-Constants::fHeightReadout);
	
	new ((*mReadoutElements)[mReadoutElements->GetEntries()]) THnSparseC(Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									     Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									     3, nBins, minPosition, maxPosition);
      
      }

      // right element: y < 0
      
      if (isFront) zMin = zMinFront;
      else         zMin = zMinBack;

      minPosition[0] = +1.*intLimitDetElem;
      minPosition[1] = lowEdgeActive;
      minPosition[2] = zMin;
      
      maxPosition[0] = +1.*extLimitDetElem;
      maxPosition[1] = supEdgeActive;
      maxPosition[2] = zMin+mThicknessActive; 
      
      new ((*mActiveElements)[mActiveElements->GetEntries()]) THnSparseC(Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									 3, nBins, minPosition, maxPosition);	
      
      minPosition[1] = lowEdgeActive-Constants::fHeightReadout;
      maxPosition[1] = lowEdgeActive;

      new ((*mReadoutElements)[mReadoutElements->GetEntries()]) THnSparseC(Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									   3, nBins, minPosition, maxPosition);

      // right element: y > 0
      
      if (supEdgeActive < 0.5*Constants::fHeightActive) {

	if (mPlaneIsOdd) {
	  if (isFront) zMin = zMinBack;
	  else         zMin = zMinFront;
	}
	
	minPosition[0] = +1.*intLimitDetElem;
	minPosition[1] = -1.*supEdgeActive;
	minPosition[2] = zMin;
	
	maxPosition[0] = +1.*extLimitDetElem;
	maxPosition[1] = -1.*lowEdgeActive;
	maxPosition[2] = zMin+mThicknessActive; 
	
	new ((*mActiveElements)[mActiveElements->GetEntries()]) THnSparseC(Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									   Form("MFTActiveElemHist_%02d%03d", mPlaneNumber, mActiveElements->GetEntries()), 
									   3, nBins, minPosition, maxPosition);	
	
	minPosition[1] = -1.*lowEdgeActive;
	maxPosition[1] = -1.*(lowEdgeActive-Constants::fHeightReadout);
	
	new ((*mReadoutElements)[mReadoutElements->GetEntries()]) THnSparseC(Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									     Form("MFTReadoutElemHist_%02d%03d", mPlaneNumber, mReadoutElements->GetEntries()), 
									     3, nBins, minPosition, maxPosition);

      }
      
    }
    
    lowEdgeActive += Constants::fHeightActive - Constants::fActiveSuperposition;
    supEdgeActive = lowEdgeActive + Constants::fHeightActive;
    isFront = !isFront;
    
  }
  
  // ------------------- support element -------------------------------------------------
  
  nBins[0] = 1;
  nBins[1] = 1;
  nBins[2] = 1;
  
  minPosition[0] = -1.*mRMaxSupport;
  minPosition[1] = -1.*mRMaxSupport;
  minPosition[2] = mZCenter - 0.5*mThicknessSupport;
  
  maxPosition[0] = +1.*mRMaxSupport;
  maxPosition[1] = +1.*mRMaxSupport;
  maxPosition[2] = mZCenter + 0.5*mThicknessSupport;
  
  new ((*mSupportElements)[mSupportElements->GetEntries()]) THnSparseC(Form("MFTSupportElemHist_%02d%03d", mPlaneNumber, mSupportElements->GetEntries()), 
								       Form("MFTSupportElemHist_%02d%03d", mPlaneNumber, mSupportElements->GetEntries()), 
								       3, nBins, minPosition, maxPosition);

  // --------------------------------------------------------------------------------------

  LOG(DEBUG1) << "CreateStructure " << Form("structure completed for MFT plane %s", GetName()) << FairLogger::endl;

  return kTRUE;
  
}

//_____________________________________________________________________________
THnSparseC* Plane::GetActiveElement(Int_t id) 
{

  if (id<0 || id>=GetNActiveElements()) return nullptr;
  else return (THnSparseC*) mActiveElements->At(id);

}

//_____________________________________________________________________________
THnSparseC* Plane::GetReadoutElement(Int_t id) 
{

  if (id<0 || id>=GetNReadoutElements()) return nullptr;
  else return (THnSparseC*) mReadoutElements->At(id);

}


//_____________________________________________________________________________
THnSparseC* Plane::GetSupportElement(Int_t id) 
{

  if (id<0 || id>=GetNSupportElements()) return nullptr;
  else return (THnSparseC*) mSupportElements->At(id);

}

//_____________________________________________________________________________
void Plane::DrawPlane(Option_t *opt) 
{

  // ------------------- "FRONT" option ------------------

  if (!strcmp(opt, "front")) {

    auto *cnv = new TCanvas("cnv", GetName(), 900, 900);
    cnv->Draw();

    auto *h = new TH2D("tmp", GetName(), 
		       1, 1.1*GetSupportElement(0)->GetAxis(0)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(0)->GetXmax(), 
		       1, 1.1*GetSupportElement(0)->GetAxis(1)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(1)->GetXmax());
    h->SetXTitle("x [cm]");
    h->SetYTitle("y [cm]");
    h->Draw();

    Info("DrawPlane","Created hist",0,0);

    auto *supportExt = new TEllipse(0.0, 0.0, mRMaxSupport, mRMaxSupport);
    auto *supportInt = new TEllipse(0.0, 0.0, mRMinSupport, mRMinSupport);
    supportExt->SetFillColor(kCyan-10);
    supportExt -> Draw("same");
    supportInt -> Draw("same");

    for (Int_t iEl=0; iEl<GetNActiveElements(); iEl++) {
      if (!IsFront(GetActiveElement(iEl))) continue;
      auto *pave = new TPave(GetActiveElement(iEl)->GetAxis(0)->GetXmin(), 
			      GetActiveElement(iEl)->GetAxis(1)->GetXmin(), 
			      GetActiveElement(iEl)->GetAxis(0)->GetXmax(), 
			      GetActiveElement(iEl)->GetAxis(1)->GetXmax(), 1);
      pave -> SetFillColor(kGreen);
      pave -> Draw("same");
    }

    for (Int_t iEl=0; iEl<GetNReadoutElements(); iEl++) {
      if (!IsFront(GetReadoutElement(iEl))) continue;
      auto *pave = new TPave(GetReadoutElement(iEl)->GetAxis(0)->GetXmin(), 
			      GetReadoutElement(iEl)->GetAxis(1)->GetXmin(), 
			      GetReadoutElement(iEl)->GetAxis(0)->GetXmax(), 
			      GetReadoutElement(iEl)->GetAxis(1)->GetXmax(), 1);
      pave -> SetFillColor(kRed);
      pave -> Draw("same");
    }

  }
    
  // ------------------- "BACK" option ------------------

  else if (!strcmp(opt, "back")) {

    auto *cnv = new TCanvas("cnv", GetName(), 900, 900);
    cnv->Draw();
    
    auto *h = new TH2D("tmp", GetName(), 
		       1, 1.1*GetSupportElement(0)->GetAxis(0)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(0)->GetXmax(), 
		       1, 1.1*GetSupportElement(0)->GetAxis(1)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(1)->GetXmax());
    h->SetXTitle("x [cm]");
    h->SetYTitle("y [cm]");
    h->Draw();

    auto *supportExt = new TEllipse(0.0, 0.0, mRMaxSupport, mRMaxSupport);
    auto *supportInt = new TEllipse(0.0, 0.0, mRMinSupport, mRMinSupport);
    supportExt -> SetFillColor(kCyan-10);
    supportExt -> Draw("same");
    supportInt -> Draw("same");

    for (Int_t iEl=0; iEl<GetNActiveElements(); iEl++) {
      if (IsFront(GetActiveElement(iEl))) continue;
      auto *pave = new TPave(GetActiveElement(iEl)->GetAxis(0)->GetXmin(), 
			      GetActiveElement(iEl)->GetAxis(1)->GetXmin(), 
			      GetActiveElement(iEl)->GetAxis(0)->GetXmax(), 
			      GetActiveElement(iEl)->GetAxis(1)->GetXmax(), 1);
      pave -> SetFillColor(kGreen);
      pave -> Draw("same");
    }

    for (Int_t iEl=0; iEl<GetNReadoutElements(); iEl++) {
      if (IsFront(GetReadoutElement(iEl))) continue;
      auto *pave = new TPave(GetReadoutElement(iEl)->GetAxis(0)->GetXmin(), 
			      GetReadoutElement(iEl)->GetAxis(1)->GetXmin(), 
			      GetReadoutElement(iEl)->GetAxis(0)->GetXmax(), 
			      GetReadoutElement(iEl)->GetAxis(1)->GetXmax(), 1);
      pave -> SetFillColor(kRed);
      pave -> Draw("same");
    }

  }

  // ------------------- "BOTH" option ------------------

  else if (!strcmp(opt, "both")) {

    auto *cnv = new TCanvas("cnv", GetName(), 900, 900);
    cnv->Draw();

    auto *h = new TH2D("tmp", GetName(), 
		       1, 1.1*GetSupportElement(0)->GetAxis(0)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(0)->GetXmax(), 
		       1, 1.1*GetSupportElement(0)->GetAxis(1)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(1)->GetXmax());
    h->SetXTitle("x [cm]");
    h->SetYTitle("y [cm]");
    h->Draw();

    auto *supportExt = new TEllipse(0.0, 0.0, mRMaxSupport, mRMaxSupport);
    auto *supportInt = new TEllipse(0.0, 0.0, mRMinSupport, mRMinSupport);
    supportExt -> SetFillColor(kCyan-10);
    supportExt -> Draw("same");
    supportInt -> Draw("same");

    for (Int_t iEl=0; iEl<GetNActiveElements(); iEl++) {
      if (IsFront(GetActiveElement(iEl)) && GetActiveElement(iEl)->GetAxis(0)->GetXmin()<0.) {
	auto *pave = new TPave(GetActiveElement(iEl)->GetAxis(0)->GetXmin(), 
				GetActiveElement(iEl)->GetAxis(1)->GetXmin(), 
				TMath::Min(GetActiveElement(iEl)->GetAxis(0)->GetXmax(), 0.),
				GetActiveElement(iEl)->GetAxis(1)->GetXmax(), 1);
	pave -> SetFillColor(kGreen);
	pave -> Draw("same");
      }
      else if (!IsFront(GetActiveElement(iEl)) && GetActiveElement(iEl)->GetAxis(0)->GetXmax()>0.) {
	auto *pave = new TPave(TMath::Max(GetActiveElement(iEl)->GetAxis(0)->GetXmin(), 0.), 
				GetActiveElement(iEl)->GetAxis(1)->GetXmin(), 
				GetActiveElement(iEl)->GetAxis(0)->GetXmax(), 
				GetActiveElement(iEl)->GetAxis(1)->GetXmax(), 1);
	pave -> SetFillColor(kGreen);
	pave -> Draw("same");
      }
    }
    
    for (Int_t iEl=0; iEl<GetNReadoutElements(); iEl++) {
      if (IsFront(GetReadoutElement(iEl)) && GetReadoutElement(iEl)->GetAxis(0)->GetXmin()<0.) {
	auto *pave = new TPave(GetReadoutElement(iEl)->GetAxis(0)->GetXmin(), 
				GetReadoutElement(iEl)->GetAxis(1)->GetXmin(), 
				TMath::Min(GetReadoutElement(iEl)->GetAxis(0)->GetXmax(), 0.), 
				GetReadoutElement(iEl)->GetAxis(1)->GetXmax(), 1);
	pave -> SetFillColor(kRed);
	pave -> Draw("same");
      }
      else if (!IsFront(GetReadoutElement(iEl)) && GetReadoutElement(iEl)->GetAxis(0)->GetXmax()>0.) {
	auto *pave = new TPave(TMath::Max(GetReadoutElement(iEl)->GetAxis(0)->GetXmin(), 0.),  
				GetReadoutElement(iEl)->GetAxis(1)->GetXmin(), 
				GetReadoutElement(iEl)->GetAxis(0)->GetXmax(), 
				GetReadoutElement(iEl)->GetAxis(1)->GetXmax(), 1);
	pave -> SetFillColor(kRed);
	pave -> Draw("same");
      }
    }
    
  }

  // ------------------- "PROFILE" option ------------------

  else if (!strcmp(opt, "profile")) {

    auto *cnv = new TCanvas("cnv", GetName(), 300, 900);
    cnv->Draw();

    auto *h = new TH2D("tmp", GetName(), 
		       1, mZCenter-0.5, mZCenter+0.5, 
		       1, 1.1*GetSupportElement(0)->GetAxis(1)->GetXmin(), 1.1*GetSupportElement(0)->GetAxis(1)->GetXmax());
    h->SetXTitle("z [cm]");
    h->SetYTitle("y [cm]");
    h->Draw();

    auto *supportExt = new TPave(GetSupportElement(0)->GetAxis(2)->GetXmin(), -mRMaxSupport, 
				  GetSupportElement(0)->GetAxis(2)->GetXmax(),  mRMaxSupport);
    auto *supportInt = new TPave(GetSupportElement(0)->GetAxis(2)->GetXmin(), -mRMinSupport, 
				  GetSupportElement(0)->GetAxis(2)->GetXmax(),  mRMinSupport);
    supportExt -> SetFillColor(kCyan-10);
    supportInt -> SetFillColor(kCyan-10);
    supportExt -> SetBorderSize(1);
    supportInt -> SetBorderSize(1);
    supportExt -> Draw("same");
    supportInt -> Draw("same");

    for (Int_t iEl=0; iEl<GetNActiveElements(); iEl++) {
      TPave * pave = nullptr;
      if (IsFront(GetActiveElement(iEl))) {
	pave = new TPave(GetActiveElement(iEl)->GetAxis(2)->GetXmax() - 
			 5*(GetActiveElement(iEl)->GetAxis(2)->GetXmax()-GetActiveElement(iEl)->GetAxis(2)->GetXmin()), 
			 GetActiveElement(iEl)->GetAxis(1)->GetXmin(), 
			 GetActiveElement(iEl)->GetAxis(2)->GetXmax(), 
			 GetActiveElement(iEl)->GetAxis(1)->GetXmax(), 1);
      }
      else {
	pave = new TPave(GetActiveElement(iEl)->GetAxis(2)->GetXmin(), 
			 GetActiveElement(iEl)->GetAxis(1)->GetXmin(), 
			 GetActiveElement(iEl)->GetAxis(2)->GetXmin() + 
			 5*(GetActiveElement(iEl)->GetAxis(2)->GetXmax()-GetActiveElement(iEl)->GetAxis(2)->GetXmin()), 
			 GetActiveElement(iEl)->GetAxis(1)->GetXmax(), 1);
      }	
      pave -> SetFillColor(kGreen);
      pave -> Draw("same");
    }
    
    for (Int_t iEl=0; iEl<GetNReadoutElements(); iEl++) {
      TPave *pave = nullptr;
      if (IsFront(GetReadoutElement(iEl))) {
	pave = new TPave(GetReadoutElement(iEl)->GetAxis(2)->GetXmax() - 
			 5*(GetReadoutElement(iEl)->GetAxis(2)->GetXmax()-GetReadoutElement(iEl)->GetAxis(2)->GetXmin()), 
			 GetReadoutElement(iEl)->GetAxis(1)->GetXmin(), 
			 GetReadoutElement(iEl)->GetAxis(2)->GetXmax(), 
			 GetReadoutElement(iEl)->GetAxis(1)->GetXmax(), 1);
      }
      else {
	pave = new TPave(GetReadoutElement(iEl)->GetAxis(2)->GetXmin(), 
			 GetReadoutElement(iEl)->GetAxis(1)->GetXmin(), 
			 GetReadoutElement(iEl)->GetAxis(2)->GetXmin() + 
			 5*(GetReadoutElement(iEl)->GetAxis(2)->GetXmax()-GetReadoutElement(iEl)->GetAxis(2)->GetXmin()), 
			 GetReadoutElement(iEl)->GetAxis(1)->GetXmax(), 1);
      }	
      pave -> SetFillColor(kRed);
      pave -> Draw("same");
    }
    
  }

}

//_____________________________________________________________________________
Int_t Plane::GetNumberOfChips(Option_t *opt) 
{

  Int_t nChips = 0;

  if (!strcmp(opt, "front")) {
    for (Int_t iEl=0; iEl<GetNActiveElements(); iEl++) {
      if (!IsFront(GetActiveElement(iEl))) continue;
      Double_t length = GetActiveElement(iEl)->GetAxis(0)->GetXmax() - GetActiveElement(iEl)->GetAxis(0)->GetXmin();
      nChips += Int_t (length/Constants::fWidthChip) + 1;
    }
  }

  else if (!strcmp(opt, "back")) {
    for (Int_t iEl=0; iEl<GetNActiveElements(); iEl++) {
      if (IsFront(GetActiveElement(iEl))) continue;
      Double_t length = GetActiveElement(iEl)->GetAxis(0)->GetXmax() - GetActiveElement(iEl)->GetAxis(0)->GetXmin();
      nChips += Int_t (length/Constants::fWidthChip) + 1;
    }
  }

  return nChips;

}

