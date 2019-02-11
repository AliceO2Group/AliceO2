#ifdef __CINT__

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedtypedefs;

#pragma link C++ class TKDInterpolatorBase+;
#pragma link C++ class TKDNodeInfo+;
#pragma link C++ class TKDNodeInfo::TKDNodeDraw+;
#pragma link C++ class TKDPDF+;
#pragma link C++ class TKDInterpolator+;
#pragma link C++ class TKDSpline+;
#pragma link C++ class TTreeDataElement+;
#pragma link C++ class TTreeStream+;
#pragma link C++ class TTreeSRedirector+;

#pragma link C++ class AliTMinuitToolkit+;
#pragma link C++ class AliNDLocalRegression+;
#pragma link C++ class AliMathBase+;

#ifdef ROOT_HAS_HTTP
#pragma link C++ class AliTreePlayer+;
#pragma link C++ class AliTreeFormulaF+;
#endif

#pragma link C++ class AliExternalInfo+;
#pragma link C++ class AliTreeTrending+;
#pragma link C++ class AliNDFormulaBrowser+;

#pragma link C++ class AliDrawStyle+;
#pragma link C++ function AliDrawStyle::TObjectApplyStyle(const char*, TGraph *, Int_t, Int_t);
#pragma link C++ function AliDrawStyle::TObjectApplyStyle(const char*, TH1 *, Int_t, Int_t);
#pragma link C++ function AliDrawStyle::TObjectApplyStyle(const char*, TF1 *, Int_t, Int_t);

#pragma link C++ class AliPainter+;
#pragma link C++ function AliPainter::SetFitter(TH1D *, std::map<TString, TString>, Int_t);
//#pragma link C++ function AliPainter::SetFitter(TH2D *, Int_t);
//#pragma link C++ function AliPainter::SetFitter(TH3D *, Int_t);
#pragma link C++ function AliPainter::SetDrawingOptions(TH1D *&, std::map<TString, TString>, Int_t);
//#pragma link C++ function AliPainter::SetDrawingOptions(TH2D *&, Int_t);
//#pragma link C++ function AliPainter::SetDrawingOptions(TH3D *&, Int_t);

#pragma link C++ class AliParser+;

#pragma link C++ class AliElasticSearchRoot++;

#pragma link C++ namespace AliFFTsmoother+;

#pragma link C++ namespace TStatToolkit;
#pragma link C++ function  TStatToolkit::LTMHisto(TH1 * , TVectorF &, Float_t);
#pragma link C++ function  TStatToolkit::LTMHisto(TH1 * , TVectorD &, Float_t);
#pragma link C++ function  TStatToolkit::LTMUnbinned(int, const float*, TVectorF & , Float_t);
#pragma link C++ function  TStatToolkit::LTMUnbinned(int, const double*, TVectorD & , Float_t);

#pragma link C++ function TStatToolkit::Reorder(int , float *, const int *);
#pragma link C++ function TStatToolkit::Reorder(int , double *, const int *);
#pragma link C++ function TStatToolkit::Reorder(int , int *, const int *);
#pragma link C++ function TStatToolkit::Reorder(int , short *, const int *);
#pragma link C++ function TStatToolkit::Reorder(int , char *, const int *);
#pragma link C++ class std::map<std::string,TTree*>+;
#pragma link C++ class std::map<std::string,TTree*>+;
#pragma link C++ class std::map<TString, TString>+;
#pragma link C++ operators std::map<TString, TString>::iterator;
#pragma link C++ class std::map<std::string,TObject*>+;
#pragma link C++ class std::map<ULong64_t,TString>+;
#pragma link C++ class std::map<ULong64_t,Int_t>+;
#pragma link C++ class std::map<ULong64_t,UInt_t>+;
#pragma link C++ class std::map<UInt_t,ULong64_t>+;
#pragma link C++ class std::map<std::string,TFormula*>+;
#pragma link C++ class std::map<std::string,TVectorF*>+;
#pragma link C++ class std::map<std::string,TVectorD*>+;
#pragma link C++ class std::map<UInt_t,THn*>+;
#pragma link C++ class std::map<Int_t,TClonesArray*>+;
#pragma link C++ class std::map<std::string,std::string>+;
#pragma link C++ class std::map<Int_t,std::string>+;
/*
// RS At the moment is not recognized by the CINT...

#pragma link C++ function TStatToolkit::GetMinMax(const Double_t*, Long64_t, double &, double &);
#pragma link C++ function TStatToolkit::GetMinMax(const Float_t*, Long64_t, double &, double &);
#pragma link C++ function TStatToolkit::GetMinMax(const Int_t*, Long64_t, double &, double &);
#pragma link C++ function TStatToolkit::GetMinMaxMean(const Double_t*, Long64_t, double &, double &, double &);
#pragma link C++ function TStatToolkit::GetMinMaxMean(const Float_t*, Long64_t, double &, double &, double &);
#pragma link C++ function TStatToolkit::GetMinMaxMean(const Int_t*, Long64_t, double &, double &, double&);

#pragma link C++ function TStatToolkit::TruncatedMean(const TH1*, TVectorF *,Float_t,Float_t,Bool_t);
#pragma link C++ function TStatToolkit::TruncatedMean(const TH1*, TVectorD *,Float_t,Float_t,Bool_t);

#pragma link C++ function TStatToolkit::LTMHisto(TH1*,TVectorF &,Float_t);
#pragma link C++ function TStatToolkit::LTMHisto(TH1*,TVectorD &,Float_t);

#pragma link C++ function TStatToolkit::LTM(TH1*,TVectorF*,Float_t,Bool_t);
#pragma link C++ function TStatToolkit::LTM(TH1*,TVectorD*,Float_t,Bool_t);


#pragma link C++ function TStatToolkit::FitGaus(TH1*,TVectorF*,TMatrixF*,Float_t,Float_t,Bool_t);
#pragma link C++ function TStatToolkit::FitGaus(TH1*,TVectorD*,TMatrixD*,Float_t,Float_t,Bool_t);

#pragma link C++ function TStatToolkit::FitGaus(Float_t*,Int_t,Float_t,Float_t,TVectorF*,TMatrixF*,Bool_t);
#pragma link C++ function TStatToolkit::FitGaus(Float_t*,Int_t,Float_t,Float_t,TVectorD*,TMatrixD*,Bool_t);

*/

#endif
