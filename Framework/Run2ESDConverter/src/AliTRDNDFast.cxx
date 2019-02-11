// Author: Daniel.Lohner@cern.ch

#include "AliTRDNDFast.h"
#include "AliLog.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TFitResult.h"
#include <iostream>
#include <fstream>

extern Double_t langaufunc(Double_t *x, Double_t *par) {

    // needs protection, parameter [3]>0!!!!!
    // needs protection, parameter [4]>0!!!!!

    //Fit parameters:
    //par[0]=Width (scale) parameter of Landau density
    //par[1]=Most Probable (MP, location) parameter of Landau density
    //par[2]=Total area (integral -inf to inf, normalization constant)
    //par[3]=Width (sigma) of convoluted Gaussian function
    //par[4]=Exponential Slope Parameter
    //
    //In the Landau distribution (represented by the CERNLIB approximation),
    //the maximum is located at x=-0.22278298 with the location parameter=0.
    //This shift is corrected within this function, so that the actual
    //maximum is identical to the MP parameter.

    // Numeric constants
    Double_t invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
    Double_t mpshift  = -0.22278298;       // Landau maximum location

    // Control constants
    Double_t np = 100.0;      // number of convolution stpdf
    Double_t sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

    // Variables
    Double_t xx;
    Double_t mpc;
    Double_t fland;
    Double_t sum = 0.0;
    Double_t xlow,xupp;
    Double_t step;
    Double_t i;

    // MP shift correction
    mpc = par[1] - mpshift * par[0];

    // Range of convolution integral
    xlow = x[0] - sc * par[3];
    xupp = x[0] + sc * par[3];

    if(xlow<0)xlow=0;
    if(xupp<xlow)cout<<"ERROR: Convolution around Negative MPV"<<endl;

    step = (xupp-xlow) / np;

    // Convolution integral of Landau and Gaussian by sum
    for(i=1.0; i<=np/2; i++) {
        xx = xlow + (i-.5) * step;
        fland = TMath::Landau(xx,mpc,par[0])*TMath::Exp(-par[4]*xx) / par[0];	// mpc taken out
        sum += fland * TMath::Gaus(x[0],xx,par[3]);

        xx = xupp - (i-.5) * step;
        fland = TMath::Landau(xx,mpc,par[0])*TMath::Exp(-par[4]*xx) / par[0];	// mpc taken out
        sum += fland * TMath::Gaus(x[0],xx,par[3]);
    }

    return (par[2] * step * sum * invsq2pi / par[3]);
}



ClassImp(AliTRDNDFast);

AliTRDNDFast::AliTRDNDFast(): TObject(),
    fNDim(0),
    fTitle(""),
    fFunc(NULL),
    fHistos(NULL),
    fPars(),
    iLangauFitOptionParameter(0)
{
    // default constructor
}

AliTRDNDFast::AliTRDNDFast(const char *name,Int_t ndim,Int_t nbins,Double_t xlow,Double_t xup): TObject(),
    fNDim(ndim),
    fTitle(name),
    fFunc(NULL),
    fHistos(NULL),
    fPars(),
    iLangauFitOptionParameter(0)
{
    Init();
    for(Int_t idim=0;idim<fNDim;idim++){
        fHistos[idim]=new TH1F(Form("%s_axis_%d",fTitle.Data(),idim),Form("%s_axis_%d",fTitle.Data(),idim),nbins,xlow,xup);
        fHistos[idim]->SetDirectory(0);
    }
}

AliTRDNDFast::AliTRDNDFast(const AliTRDNDFast &ref) : TObject(ref),
    fNDim(ref.fNDim),
    fTitle(ref.fTitle),
    fFunc(NULL),
    fHistos(NULL),
    fPars(),
    iLangauFitOptionParameter(ref.iLangauFitOptionParameter)
{
    Cleanup();
    Init();
    for(Int_t idim=0;idim<fNDim;idim++){
        fHistos[idim]=(TH1F*)ref.fHistos[idim]->Clone(Form("%s_axis_%d",GetName(),idim));
        fHistos[idim]->SetDirectory(0);
        for(Int_t ipar=0;ipar<kNpar;ipar++)fPars[idim][ipar]=ref.fPars[idim][ipar];
    }
}

AliTRDNDFast &AliTRDNDFast::operator=(const AliTRDNDFast &ref){
    //
    // Assignment operator
    //
    if(this != &ref){
        TObject::operator=(ref);
        fNDim=ref.fNDim;
        fTitle=ref.fTitle;
        fFunc = ref.fFunc;
        iLangauFitOptionParameter=ref.iLangauFitOptionParameter;
        for(Int_t idim=0;idim<fNDim;idim++){
            fHistos[idim]=(TH1F*)ref.fHistos[idim]->Clone(Form("%s_axis_%d",GetName(),idim));
            fHistos[idim]->SetDirectory(0);
            for(Int_t ipar=0;ipar<kNpar;ipar++)fPars[idim][ipar]=ref.fPars[idim][ipar];
        }
    }
    return *this;
}

AliTRDNDFast::~AliTRDNDFast(){
    Cleanup();

}

void AliTRDNDFast::Init(){

    for(Int_t ipar=0;ipar<kNpar;ipar++)fPars[ipar].Set(fNDim);
    fFunc=new TF1*[fNDim];
    fHistos=new TH1F*[fNDim];
    for(Int_t idim=0;idim<fNDim;idim++){
        fHistos[idim]=NULL;
        for(Int_t ipar=0;ipar<kNpar;ipar++)fPars[ipar].SetAt(0,idim);
        fFunc[idim]=NULL;
    }
}

void AliTRDNDFast::Cleanup(){
    if(fHistos){
        for(Int_t idim=0;idim<fNDim;idim++){
            if(fHistos[idim]){
                delete fHistos[idim];
                fHistos[idim]=NULL;
            }
        }
        delete[] fHistos;
        fHistos=NULL;
    }
    for(Int_t ipar=0;ipar<kNpar;ipar++){
        fPars[ipar].Reset();
    }
    if(fFunc){
        for(Int_t idim=0;idim<fNDim;idim++){
            if(fFunc[idim]){
                delete fFunc[idim];
                fFunc[idim]=NULL;
            }
        }
        delete[] fFunc;
        fFunc=NULL;
    }
}

void AliTRDNDFast::PrintPars(){
    for(Int_t idim=0;idim<fNDim;idim++){
        for(Int_t ipar=0;ipar<kNpar;ipar++)cout<<idim<<" "<<ipar<<" "<<GetParam(idim,ipar)<<endl;
    }
}

Int_t AliTRDNDFast::GetFitOptionParameter(){
    return iLangauFitOptionParameter;
}

void AliTRDNDFast::SetFitOptionParameter(Int_t iFitParameter){
    iLangauFitOptionParameter=iFitParameter;
}

void AliTRDNDFast::ScaleLangauFun(TF1 *func,Double_t mpv){

    Double_t start[kNpar],low[kNpar],high[kNpar];
    for(Int_t ii=0;ii<kNpar;ii++){
        start[ii]=func->GetParameter(ii);
        func->GetParLimits(ii,low[ii],high[ii]);
    }

    Double_t scalefactor=mpv/start[1];

    for(Int_t ii=0;ii<kNpar;ii++){
        Double_t scale=1;
        if(ii==0||ii==1||ii==3)scale=scalefactor;
        if(ii==4)scale=1./scalefactor;
        start[ii]*=scale;
        low[ii]*=scale;
        high[ii]*=scale;
        func->SetParLimits(ii, low[ii], high[ii]);
    }
    func->SetParameters(start);
}

//---------------------------------------------------------------
//---------------------------------------------------------------
TF1 *AliTRDNDFast::GetLangauFun(TString funcname,Double_t range[2],Double_t scalefactor){

    Double_t start[kNpar] = {120, 1000, 1, 100, 1e-5};
    Double_t low[kNpar] = {10, 300, 0.01, 1, 0.};
    Double_t high[kNpar] = {1000, 3000, 100, 1000, 1.};

    TF1 *hlandauE=new TF1(funcname.Data(),langaufunc,0,range[1],kNpar);
    hlandauE->SetParameters(start);
    hlandauE->SetParNames("Width","MP","Area","GSigma","delta");
    for (Int_t i=0; i<kNpar; i++) {
        hlandauE->SetParLimits(i, low[i], high[i]);
    }
    if (iLangauFitOptionParameter==1){
        hlandauE->FixParameter(4,0);
    }
    if(scalefactor!=1){ScaleLangauFun(hlandauE,scalefactor*start[1]);}

    return hlandauE;
}

TF1 *AliTRDNDFast::FitLandau(TString name,TH1F *htemp,Double_t range[2],TString option){
    //cout<<"Started Fit Landau routine of name: "<<name.Data()<<endl;
    //ofstream FileToDebugFit;
    //FileToDebugFit.open("FileToDebugFit.txt",std::ios::app);
    TF1 *fitlandau1D=GetLangauFun(name,range);
    TF1 fit("land","landau");
    Double_t max=htemp->GetXaxis()->GetBinCenter(htemp->GetMaximumBin());
    fit.SetParameter(1,max);
    fit.SetParLimits(1,0,htemp->GetXaxis()->GetXmax());
    fit.SetParameter(2,0.3*max); // MPV may never become negative!!!!!
    htemp->Fit("land",option.Data(),"",range[0],range[1]);
    ScaleLangauFun(fitlandau1D,fit.GetParameter(1));
    //cout<<"LanDau Fit performed and LanGau scaled"<<endl;
    htemp->Fit(fitlandau1D,option.Data(),"",range[0],range[1]); // range for references
    //TFitResultPtr FitRes=htemp->Fit(fitlandau1D,option.Data(),"",range[0],range[1]); // range for references
    /*cout<<"got Fit result"<<endl;
    FitRes->GetCovarianceMatrix();
    TMatrixDSym TMatrCovFitRes=FitRes->GetCovarianceMatrix();
    TMatrixDSym TMatrCorFitRes=FitRes->GetCorrelationMatrix();
    Double_t dChi2=FitRes->Chi2();
    Double_t dNDF=FitRes->Ndf();
    Double_t dChiDivNdf=dChi2/dNDF;
    //cout<<"got Chi2()"<<endl;
    FileToDebugFit<<" Chi2 "<<dChiDivNdf<<endl;
    FileToDebugFit<<" CorrMatrix:"<<endl;
    for (Int_t i=0; i<kNpar; i++){
        for (Int_t j=0; j<kNpar; j++){
            FileToDebugFit<<TMatrCorFitRes(i,j)<<" ";
        }
        FileToDebugFit<<endl;
    }
    //FitRes->Print("V");
    FileToDebugFit.close();
    FileToDebugFit.open("FileToDebugFit2.txt",std::ios::app);
    FileToDebugFit<<dChiDivNdf<<" ";
    FileToDebugFit.close();*/
    return fitlandau1D;
}

void AliTRDNDFast::BuildHistos(){

    for(Int_t idim=0;idim<fNDim;idim++){
        fHistos[idim]->Reset();
        // Fill Histo
        Double_t pars[kNpar];
        for(Int_t ipar=0;ipar<kNpar;ipar++)pars[ipar]=GetParam(idim,ipar);
        // Also Fill overflow bins!!!
        for(Int_t ii=1;ii<=fHistos[idim]->GetNbinsX()+1;ii++){
            Double_t xval=fHistos[idim]->GetXaxis()->GetBinCenter(ii);
            Double_t val=langaufunc(&xval,&pars[0]);
            //Double_t val=fFunc[idim]->Eval(xval);
            fHistos[idim]->SetBinContent(ii,val);
        }
        // Normalization
        if(fHistos[idim]->Integral(1,fHistos[idim]->GetNbinsX(),"width")!=0) fHistos[idim]->Scale(1./fHistos[idim]->Integral(1,fHistos[idim]->GetNbinsX(),"width"));
    }
}

void AliTRDNDFast::Build(Double_t **pars){
    // pars[ndim][npar]
    for(Int_t idim=0;idim<fNDim;idim++){
        for(Int_t ipar=0;ipar<kNpar;ipar++){
            fPars[ipar].SetAt(pars[idim][ipar],idim);
        }
    }
    BuildHistos();
}

void AliTRDNDFast::Build(TH1F **hdEdx,TString path){

    Double_t range[2];

    TCanvas *CANV=new TCanvas("a","a");
    if(fNDim==2)CANV->Divide(2,1);
    if(fNDim==3)CANV->Divide(2,2);
    if(fNDim>6)CANV->Divide(3,3);
    // Fit and Extract Parameters
    for(Int_t idim=0;idim<fNDim;idim++){
        CANV->cd(idim+1);
        gPad->SetLogy();
        range[0]=hdEdx[idim]->GetXaxis()->GetXmin();
        range[1]=hdEdx[idim]->GetXaxis()->GetXmax();
        //range[1]=8000;
        // Norm Histogram

        if(hdEdx[idim]->Integral(1,hdEdx[idim]->GetNbinsX(),"width")!=0) hdEdx[idim]->Scale(1./hdEdx[idim]->Integral(1,hdEdx[idim]->GetNbinsX(),"width"));
        // Fit Histogram
        fFunc[idim]=FitLandau(Form("fit%d",idim),hdEdx[idim],range,"RMBSQ0");
        // Norm Landau
        if(fFunc[idim]->Integral(range[0],range[1])!=0.0) fFunc[idim]->SetParameter(2,fFunc[idim]->GetParameter(2)/fFunc[idim]->Integral(range[0],range[1]));
        else {
            fFunc[idim]->SetParameter(2,fFunc[idim]->GetParameter(2));
        }
        hdEdx[idim]->DrawCopy();
        fFunc[idim]->DrawCopy("same");
        // Set Pars
        for(Int_t ipar=0;ipar<kNpar;ipar++){
            AliDebug(3,Form("parameters: %f %f %f %i %i \n",fFunc[idim]->GetParameter(ipar),fFunc[idim]->GetParError(ipar),fFunc[idim]->GetChisquare(),fFunc[idim]->GetNDF(),idim));
            fPars[ipar].SetAt(fFunc[idim]->GetParameter(ipar),idim);
        }
    }
    if(path!="")CANV->Print(Form("%s/%s_Build.pdf",path.Data(),fTitle.Data()));
    delete CANV;

    BuildHistos();
}

Double_t AliTRDNDFast::Eval(Double_t *point) const{
    Double_t val=1;
    for(Int_t idim=0;idim<fNDim;idim++){
        Int_t bin=fHistos[idim]->GetXaxis()->FindBin(point[idim]);
        val*=fHistos[idim]->GetBinContent(bin);
    }
    return val;
}

void AliTRDNDFast::Random(Double_t *point) const{
    for(Int_t idim=0;idim<fNDim;idim++){
        point[idim]=fHistos[idim]->GetRandom();
    }
}

void AliTRDNDFast::Random(Double_t *point,AliTRDNDFast *nd0,AliTRDNDFast *nd1,Double_t w0,Double_t w1){
    for(Int_t idim=0;idim<nd0->fNDim;idim++){
        point[idim]=GetRandomInterpolation(nd0->fHistos[idim],nd1->fHistos[idim],w0,w1);
    }
}

Int_t AliTRDNDFast::BinarySearchInterpolation(Int_t start,Int_t end,Double_t *a0,Double_t *a1,Double_t w0,Double_t w1,Double_t val){

    if((end-start)==1)return start;
    Int_t mid=Int_t((end+start)/2);
    Double_t valmid=(w0*a0[mid]+w1*a1[mid])/(w0+w1);
    if(val>=valmid)return BinarySearchInterpolation(mid,end,a0,a1,w0,w1,val);
    return BinarySearchInterpolation(start,mid,a0,a1,w0,w1,val);
}

Double_t AliTRDNDFast::GetRandomInterpolation(TH1F *hist0,TH1F *hist1,Double_t w0,Double_t w1){

    // Draw Random Variable
    Double_t rand=gRandom->Rndm();

    // Get Integral Arrays
    Double_t *integral0=hist0->GetIntegral();
    Double_t *integral1=hist1->GetIntegral();

    Int_t nbinsX=hist0->GetNbinsX();

    Int_t ibin=BinarySearchInterpolation(1,nbinsX+1,integral0,integral1,w0,w1,rand);
    return hist0->GetXaxis()->GetBinCenter(ibin);
}



