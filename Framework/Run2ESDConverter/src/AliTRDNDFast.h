// Author: Daniel.Lohner@cern.ch

#ifndef ALIROOT_AliTRDNDFast
#define ALIROOT_AliTRDNDFast

#ifndef ROOT_TH1
#include "TH1F.h"
#endif
#ifndef ROOT_TArrayF
#include "TArrayF.h"
#endif
#ifndef ROOT_TF2
#include "TF1.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TRandom
#include "TRandom.h"
#endif

using namespace std;

extern Double_t langaufun(Double_t *x,Double_t *par);

class AliTRDNDFast : public TObject {

public:
    static const Int_t kNpar = 5;

    AliTRDNDFast();
    AliTRDNDFast(const char *name,Int_t ndim,Int_t nbins,Double_t xlow,Double_t xup);
    AliTRDNDFast(const AliTRDNDFast&);
    AliTRDNDFast &operator=(const AliTRDNDFast &ref);
    virtual ~AliTRDNDFast();
    
    TF1 *FitLandau(TString name,TH1F *htemp,Double_t range[2],TString option);

    void Build(TH1F **hdEdx,TString path="");
    void Build(Double_t **pars);
    Double_t Eval(Double_t *point) const;
    void Random(Double_t *point) const;
    Int_t GetNDim(){return fNDim;};
    Double_t GetParam(Int_t dim,Int_t par){if((dim>=0)&&(dim<fNDim)&&(par>=0)&&(par<kNpar)){return fPars[par].GetAt(dim);}else{return 0;}};
    void PrintPars();
    static void Random(Double_t *point,AliTRDNDFast *nd0,AliTRDNDFast *nd1,Double_t w0,Double_t w1);
    Int_t GetFitOptionParameter();
    void SetFitOptionParameter(Int_t iFitParameter=0);

private:

    void ScaleLangauFun(TF1 *func,Double_t mpv);
    TF1 *GetLangauFun(TString funcname,Double_t range[2],Double_t scalefactor=1);
    void BuildHistos();
    void Init();
    void Cleanup();
    
    static Int_t BinarySearchInterpolation(Int_t start,Int_t end,Double_t *a0,Double_t *a1,Double_t w0,Double_t w1,Double_t val);
    static Double_t GetRandomInterpolation(TH1F *hist0,TH1F *hist1,Double_t w0,Double_t w1);
    Int_t fNDim; // Dimensions
    TString fTitle; //title
    TF1 **fFunc; //! functions, do not store
    TH1F **fHistos; //[fNDim] Histograms
    TArrayF fPars[kNpar]; // parameters
    Int_t iLangauFitOptionParameter;//0 Use Standard, 1 dont use Exp
    ClassDef(AliTRDNDFast,3)  //Fast TRD ND class
};

#endif
