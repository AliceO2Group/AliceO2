/*
Author: Vytautas Vislavicius
Extention of Generic Flow (https://arxiv.org/abs/1312.3572)
*/
#include "GenericFramework/GFWWeights.h"
#include "TMath.h"
GFWWeights::GFWWeights():
  fDataFilled(kFALSE),
  fMCFilled(kFALSE),
  fW_data(0),
  fW_mcrec(0),
  fW_mcgen(0),
  fEffInt(0),
  fIntEff(0),
  fAccInt(0),
  fNbinsPt(0),
  fbinsPt(0)
{
};
GFWWeights::~GFWWeights()
{
  delete fW_data;
  delete fW_mcrec;
  delete fW_mcgen;
  delete fEffInt;
  delete fIntEff;
  delete fAccInt;
  if(fbinsPt) delete [] fbinsPt;
};
void GFWWeights::SetPtBins(Int_t Nbins, Double_t *bins) {
  if(fbinsPt) delete [] fbinsPt;
  fNbinsPt = Nbins;
  fbinsPt = new Double_t[fNbinsPt+1];
  for(Int_t i=0;i<=fNbinsPt;++i) fbinsPt[i] = bins[i];
};
void GFWWeights::Init(Bool_t AddData, Bool_t AddMC)
{
  // fDataFilled = kFALSE;
  // fMCFilled = kFALSE;
  if(!fbinsPt) { //If pT bins not initialized, set to default (-1 to 1e6) to accept everything
    fNbinsPt=1;
    fbinsPt = new Double_t[2];
    fbinsPt[0] = -1;
    fbinsPt[1] = 1e6;
  };
  if(AddData) {
    fW_data = new TObjArray();
    fW_data->SetName("GFWWeights_Data");
    fW_data->SetOwner(kTRUE);
    const char *tnd = GetBinName(0,0,Form("data_%s",this->GetName()));
    fW_data->Add(new TH3D(tnd,";#varphi;#eta;v_{z}",60,0,TMath::TwoPi(),64,-1.6,1.6,40,-10,10));
    fDataFilled = kTRUE;
  };
  if(AddMC) {
    fW_mcrec = new TObjArray();
    fW_mcrec->SetName("GFWWeights_MCRec");
    fW_mcgen = new TObjArray();
    fW_mcgen->SetName("GFWWeights_MCGen");
    fW_mcrec->SetOwner(kTRUE);
    fW_mcgen->SetOwner(kTRUE);
    const char *tnr = GetBinName(0,0,"mcrec"); //all integrated over cent. anyway
    const char *tng = GetBinName(0,0,"mcgen"); //all integrated over cent. anyway
    fW_mcrec->Add(new TH3D(tnr,";#it{p}_{T};#eta;v_{z}",fNbinsPt,0,20,64,-1.6,1.6,40,-10,10));
    fW_mcgen->Add(new TH3D(tng,";#it{p}_{T};#eta;v_{z}",fNbinsPt,0,20,64,-1.6,1.6,40,-10,10));
    ((TH3D*)fW_mcrec->At(fW_mcrec->GetEntries()-1))->GetXaxis()->Set(fNbinsPt,fbinsPt);
    ((TH3D*)fW_mcgen->At(fW_mcgen->GetEntries()-1))->GetXaxis()->Set(fNbinsPt,fbinsPt);
    fMCFilled = kTRUE;
  };
};

void GFWWeights::Fill(Double_t phi, Double_t eta, Double_t vz, Double_t pt, Double_t cent, Int_t htype, Double_t weight) {
  TObjArray *tar=0;
  const char *pf="";
  if(htype==0) { tar = fW_data; pf = Form("data_%s",this->GetName()); };
  if(htype==1) { tar = fW_mcrec; pf = "mcrec"; };
  if(htype==2) { tar = fW_mcgen; pf = "mcgen"; };
  if(!tar) return;
  TH3D *th3 = (TH3D*)tar->FindObject(GetBinName(0,0,pf)); //pT bin 0, V0M bin 0, since all integrated
  if(!th3) {
    if(!htype) tar->Add(new TH3D(GetBinName(0,0,pf),";#varphi;#eta;v_{z}",60,0,TMath::TwoPi(),64,-1.6,1.6,40,-10,10)); //0,0 since all integrated
    th3 = (TH3D*)tar->At(tar->GetEntries()-1);
  };
  th3->Fill(htype?pt:phi,eta,vz, weight);
};
Double_t GFWWeights::GetWeight(Double_t phi, Double_t eta, Double_t vz, Double_t pt, Double_t cent, Int_t htype) {
  TObjArray *tar=0;
  const char *pf="";
  if(htype==0) { tar = fW_data; pf = "data"; };
  if(htype==1) { tar = fW_mcrec; pf = "mcrec"; };
  if(htype==2) { tar = fW_mcgen; pf = "mcgen"; };
  if(!tar) return 1;
  TH3D *th3 = (TH3D*)tar->FindObject(GetBinName(0,0,pf));
  if(!th3) return 1;//-1;
  Int_t xind = th3->GetXaxis()->FindBin(htype?pt:phi);
  Int_t etaind = th3->GetYaxis()->FindBin(eta);
  Int_t vzind = th3->GetZaxis()->FindBin(vz);
  Double_t weight = th3->GetBinContent(xind, etaind, vzind);
  if(weight!=0) return 1./weight;
  return 1;
};
Double_t GFWWeights::GetNUA(Double_t phi, Double_t eta, Double_t vz) {
  if(!fAccInt) CreateNUA();
  Int_t xind = fAccInt->GetXaxis()->FindBin(phi);
  Int_t etaind = fAccInt->GetYaxis()->FindBin(eta);
  Int_t vzind = fAccInt->GetZaxis()->FindBin(vz);
  Double_t weight = fAccInt->GetBinContent(xind, etaind, vzind);
  if(weight!=0) return 1./weight;
  return 1;
}
Double_t GFWWeights::GetNUE(Double_t pt, Double_t eta, Double_t vz) {
  if(!fEffInt) CreateNUE();
  Int_t xind = fEffInt->GetXaxis()->FindBin(pt);
  Int_t etaind = fEffInt->GetYaxis()->FindBin(eta);
  Int_t vzind = fEffInt->GetZaxis()->FindBin(vz);
  Double_t weight = fEffInt->GetBinContent(xind, etaind, vzind);
  if(weight!=0) return 1./weight;
  return 1;
}
Double_t GFWWeights::FindMax(TH3D *inh, Int_t &ix, Int_t &iy, Int_t &iz) {
  Double_t maxv=inh->GetBinContent(1,1,1);
  for(Int_t i=1;i<=inh->GetNbinsX();i++)
    for(Int_t j=1;j<=inh->GetNbinsY();j++)
      for(Int_t k=1;k<=inh->GetNbinsZ();k++)
	if(inh->GetBinContent(i,j,k)>maxv) {
	  ix=i;
	  iy=j;
	  iz=k;
	  maxv=inh->GetBinContent(i,j,k);
	};
  return maxv;
};
void GFWWeights::MCToEfficiency() {
  if(fW_mcgen->GetEntries()<1) {
    printf("MC gen. array empty. This is probably because effs. have been calculated and the generated particle histograms have been cleared out!\n");
    return;
  };
  for(Int_t i=0;i<fW_mcrec->GetEntries();i++) {
    TH3D *hr = (TH3D*)fW_mcrec->At(i);
    TH3D *hg = (TH3D*)fW_mcgen->At(i);
    hr->Sumw2();
    hg->Sumw2();
    hr->Divide(hg);
  };
  fW_mcgen->Clear();
};
void GFWWeights::RebinNUA(Int_t nX, Int_t nY, Int_t nZ) {
  if(fW_data->GetEntries()<1) return;
  for(Int_t i=0;i<fW_data->GetEntries();i++) {
    ((TH3D*)fW_data->At(i))->RebinX(nX);
    ((TH3D*)fW_data->At(i))->RebinY(nY);
    ((TH3D*)fW_data->At(i))->RebinZ(nZ);
  };
};
void GFWWeights::CreateNUA(Bool_t IntegrateOverCentAndPt) {
  if(!IntegrateOverCentAndPt) {
    printf("Method is outdated! NUA is integrated over centrality and pT. Quit now, or the behaviour will be bad\n");
    return;
  };
  TH3D *h3;
  TH1D *h1;
  if(fW_data->GetEntries()<1) return;
  if(IntegrateOverCentAndPt) {
    if(fAccInt) delete fAccInt;
    fAccInt = (TH3D*)fW_data->At(0)->Clone("IntegratedAcceptance");
    //fAccInt->RebinY(2); this is rebinned already during the postprocessing
    //fAccInt->RebinZ(5);
    fAccInt->Sumw2();
    for(Int_t etai=1;etai<=fAccInt->GetNbinsY();etai++) {
      fAccInt->GetYaxis()->SetRange(etai,etai);
      if(fAccInt->Integral()<1) continue;
      for(Int_t vzi=1;vzi<=fAccInt->GetNbinsZ();vzi++) {
    	fAccInt->GetZaxis()->SetRange(vzi,vzi);
    	if(fAccInt->Integral()<1) continue;
    	h1 = (TH1D*)fAccInt->Project3D("x");
    	Double_t maxv = h1->GetMaximum();
    	for(Int_t phii=1;phii<=h1->GetNbinsX();phii++) {
        fAccInt->SetBinContent(phii,etai,vzi,fAccInt->GetBinContent(phii,etai,vzi)/maxv);
        fAccInt->SetBinError(phii,etai,vzi,fAccInt->GetBinError(phii,etai,vzi)/maxv);
      };
    	delete h1;
      };
      fAccInt->GetZaxis()->SetRange(1,fAccInt->GetNbinsZ());
    };
    fAccInt->GetYaxis()->SetRange(1,fAccInt->GetNbinsY());
    return;
  };
};
TH1D *GFWWeights::GetdNdPhi() {
  TH3D *temph = (TH3D*)fW_data->At(0)->Clone("tempH3");
  TH1D *reth = (TH1D*)temph->Project3D("x");
  reth->SetName("RetHist");
  delete temph;
  Double_t max = reth->GetMaximum();
  if(max==0) return 0;
  for(Int_t phi=1; phi<=reth->GetNbinsX(); phi++) {
    if(reth->GetBinContent(phi)==0) continue;
    reth->SetBinContent(phi,reth->GetBinContent(phi)/max);
    reth->SetBinError(phi,reth->GetBinError(phi)/max);
  }
  return reth;
}
void GFWWeights::CreateNUE(Bool_t IntegrateOverCentrality) {
  if(!IntegrateOverCentrality) {
    printf("Method is outdated! NUE is integrated over centrality. Quit now, or the behaviour will be bad\n");
    return;
  };
  TH3D *num=0;
  TH3D *den=0;
  if(fW_mcrec->GetEntries()<1 || fW_mcgen->GetEntries()<1) return;
  if(IntegrateOverCentrality) {
    num=(TH3D*)fW_mcrec->At(0);//->Clone(Form("temp_%s",fW_mcrec->At(0)->GetName()));
    den=(TH3D*)fW_mcgen->At(0);//->Clone(Form("temp_%s",fW_mcgen->At(0)->GetName()));
    num->Sumw2();
    den->Sumw2();
    num->RebinY(2);
    den->RebinY(2);
    num->RebinZ(5);
    den->RebinZ(5);
    fEffInt = (TH3D*)num->Clone("Efficiency_Integrated");
    fEffInt->Divide(den);
    return;
  };
};
void GFWWeights::ReadAndMerge(TString filelinks, TString listName, Bool_t addData, Bool_t addRec, Bool_t addGen) {
  FILE *flist = fopen(filelinks.Data(),"r");
  char str[150];
  Int_t nFiles=0;
  while(fscanf(flist,"%s\n",str)==1) nFiles++;
  rewind(flist);
  if(nFiles==0) {
    printf("No files to read!\n");
    return;
  };
  if(!fW_data && addData) {
    fW_data = new TObjArray();
    fW_data->SetName("Weights_Data");
    fW_data->SetOwner(kTRUE);
  };
  if(!fW_mcrec && addRec) {
    fW_mcrec = new TObjArray();
    fW_mcrec->SetName("Weights_MCRec");
    fW_mcrec->SetOwner(kTRUE);
  };
  if(!fW_mcgen && addGen) {
    fW_mcgen = new TObjArray();
    fW_mcgen->SetName("Weights_MCGen");
    fW_mcgen->SetOwner(kTRUE);
  };
  // fDataFilled = kFALSE;
  // fMCFilled = kFALSE;
  TFile *tf=0;
  for(Int_t i=0;i<nFiles;i++) {
    Int_t trash = fscanf(flist,"%s\n",str);
    tf = new TFile(str,"READ");
    if(tf->IsZombie()) {
      printf("Could not open file %s!\n",str);
      tf->Close();
      continue;
    };
    TList *tl = (TList*)tf->Get(listName.Data());
    GFWWeights *tw = (GFWWeights*)tl->FindObject(this->GetName());
    if(!tw) {
      printf("Could not fetch weights object from %s\n",str);
      tf->Close();
      continue;
    };
    if(addData) AddArray(fW_data,tw->GetDataArray());
    if(addRec) AddArray(fW_mcrec,tw->GetRecArray());
    if(addGen) AddArray(fW_mcgen,tw->GetGenArray());
    tf->Close();
    delete tw;
  };
};
void GFWWeights::AddArray(TObjArray *targ, TObjArray *sour) {
  if(!sour) {
    printf("Source array does not exist!\n");
    return;
  };
  for(Int_t i=0;i<sour->GetEntries();i++) {
    TH3D *sourh = (TH3D*)sour->At(i);
    TH3D *targh = (TH3D*)targ->FindObject(sourh->GetName());
    if(!targh) {
      targh = (TH3D*)sourh->Clone(sourh->GetName());
      targh->SetDirectory(0);
      targ->Add(targh);
    } else
      targh->Add(sourh);
  };
};
void GFWWeights::OverwriteNUA() {
  if(!fAccInt) CreateNUA();
  TString ts(fW_data->At(0)->GetName());
  TH3D *trash = (TH3D*)fW_data->RemoveAt(0);
  delete trash;
  fW_data->Add((TH3D*)fAccInt->Clone(ts.Data()));
  delete fAccInt;
}
Long64_t GFWWeights::Merge(TCollection *collist) {
  Long64_t nmerged=0;
  if(!fW_data) {
    fW_data = new TObjArray();
    fW_data->SetName("Weights_Data");
    fW_data->SetOwner(kTRUE);
  };
  if(!fW_mcrec) {
    fW_mcrec = new TObjArray();
    fW_mcrec->SetName("Weights_MCRec");
    fW_mcrec->SetOwner(kTRUE);
  };
  if(!fW_mcgen) {
    fW_mcgen = new TObjArray();
    fW_mcgen->SetName("Weights_MCGen");
    fW_mcgen->SetOwner(kTRUE);
  };
  GFWWeights *l_w = 0;
  TIter all_w(collist);
  while (l_w = ((GFWWeights*) all_w())) {
    AddArray(fW_data,l_w->GetDataArray());
    AddArray(fW_mcrec,l_w->GetRecArray());
    AddArray(fW_mcgen,l_w->GetGenArray());
    nmerged++;
  };
  return nmerged;
};
TH1D *GFWWeights::GetIntegratedEfficiencyHist() {
  if(!fW_mcgen) { printf("MCGen array does not exist!\n"); return 0; };
  if(!fW_mcrec) { printf("MCRec array does not exist!\n"); return 0; };
  if(!fW_mcgen->GetEntries()) { printf("MCGen array is empty!\n"); return 0; };
  if(!fW_mcrec->GetEntries()) { printf("MCRec array is empty!\n"); return 0; };
  TH3D *num = (TH3D*)fW_mcrec->At(0)->Clone("Numerator");
  for(Int_t i=1;i<fW_mcrec->GetEntries();i++) num->Add((TH3D*)fW_mcrec->At(i));
  TH3D *den = (TH3D*)fW_mcgen->At(0)->Clone("Denominator");
  for(Int_t i=1;i<fW_mcgen->GetEntries();i++) den->Add((TH3D*)fW_mcgen->At(i));
  TH1D *num1d = (TH1D*)num->Project3D("x");
  num1d->SetName("retHist");
  num1d->Sumw2();
  TH1D *den1d = (TH1D*)den->Project3D("x");
  den1d->Sumw2();
  num1d->Divide(den1d);
  delete num;
  delete den;
  delete den1d;
  return num1d;
}
Bool_t GFWWeights::CalculateIntegratedEff() {
  if(fIntEff) delete fIntEff;
  fIntEff = GetIntegratedEfficiencyHist();
  if(!fIntEff) { return kFALSE; };
  fIntEff->SetName("IntegratedEfficiency");
  return kTRUE;
}
Double_t GFWWeights::GetIntegratedEfficiency(Double_t pt) {
  if(!fIntEff) if(!CalculateIntegratedEff()) return 0;
  return fIntEff->GetBinContent(fIntEff->FindBin(pt));
}
TH1D *GFWWeights::GetEfficiency(Double_t etamin, Double_t etamax, Double_t vzmin, Double_t vzmax) {
  TH3D *num = (TH3D*)fW_mcrec->At(0)->Clone("Numerator");
  for(Int_t i=1;i<fW_mcrec->GetEntries();i++) num->Add((TH3D*)fW_mcrec->At(i));
  TH3D *den = (TH3D*)fW_mcgen->At(0)->Clone("Denominator");
  for(Int_t i=1;i<fW_mcgen->GetEntries();i++) den->Add((TH3D*)fW_mcgen->At(i));
  Int_t eb1 = num->GetYaxis()->FindBin(etamin+1e-6);
  Int_t eb2 = num->GetYaxis()->FindBin(etamax-1e-6);
  Int_t vz1 = num->GetZaxis()->FindBin(vzmin+1e-6);
  Int_t vz2 = num->GetZaxis()->FindBin(vzmax-1e-6);
  num->GetYaxis()->SetRange(eb1,eb2);
  num->GetZaxis()->SetRange(vz1,vz2);
  den->GetYaxis()->SetRange(eb1,eb2);
  den->GetZaxis()->SetRange(vz1,vz2);
  TH1D *num1d = (TH1D*)num->Project3D("x");
  TH1D *den1d = (TH1D*)den->Project3D("x");
  delete num;
  delete den;
  num1d->Sumw2();
  den1d->Sumw2();
  num1d->Divide(den1d);
  delete den1d;
  return num1d;
}