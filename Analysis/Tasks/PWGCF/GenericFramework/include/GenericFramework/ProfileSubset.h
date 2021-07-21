/*
Author: Vytautas Vislavicius
Extention of Generic Flow (https://arxiv.org/abs/1312.3572)
*/
#ifndef ProfileSubset__H
#define ProfileSubset__H
//Helper function to select a subrange of a TProfile
#include "TProfile.h"
#include "TProfile2D.h"
#include "TError.h"

class ProfileSubset : public TProfile2D
{
 public:
  ProfileSubset(TProfile2D& inpf) : TProfile2D(inpf){};
  ~ProfileSubset(){};
  TProfile* GetSubset(Bool_t onx, const char* name, Int_t fb, Int_t lb, Int_t l_nbins = 0, Double_t* l_binarray = 0);
  void OverrideBinContent(Double_t x, Double_t y, Double_t x2, Double_t y2, Double_t val);
  void OverrideBinContent(Double_t x, Double_t y, Double_t x2, Double_t y2, TProfile2D* sourceProf);
  Bool_t OverrideBinsWithZero(Int_t xb1, Int_t yb1, Int_t xb2, Int_t yb2);

  ClassDef(ProfileSubset, 2);
};
#endif