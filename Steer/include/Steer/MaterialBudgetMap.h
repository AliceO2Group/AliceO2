// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 *  Created on: March 17, 2022
 *      Author: amorsch
 */

#ifndef MATERIALBUDGETMAP_H
#define MATERIALBUDGETMAP_H
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                                                                           //
//    Utility class to compute and draw Radiation Length Map                 //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

class TH2F;
namespace o2
{
namespace steer
{
class MaterialBudgetMap
{
 public:
  MaterialBudgetMap();
  MaterialBudgetMap(const char* title, Int_t mode, Int_t nc1, Float_t c1min,
                    Float_t c1max, Int_t nphi, Float_t phimin,
                    Float_t phimax, Float_t rmin, Float_t rmax, Float_t zmax);
  ~MaterialBudgetMap();
  void Stepping();
  void BeginEvent();
  void FinishPrimary(Float_t c1, Float_t c2);
  void FinishEvent();

 private:
  Int_t mMode;      //! mode
  Float_t mTotRadl; //! Total Radiation length
  Float_t mTotAbso; //! Total absorption length
  Float_t mTotGcm2; //! Total g/cm2 traversed
  TH2F* mHistRadl;  //! Radiation length map
  TH2F* mHistAbso;  //! Interaction length map
  TH2F* mHistGcm2;  //! g/cm2 length map
  TH2F* mHistReta;  //! Radiation length map as a function of eta
  TH2F* mRZR;       //! Radiation lenghts at (R.Z)
  TH2F* mRZA;       //! Absorbtion lengths at (R,Z)
  TH2F* mRZG;       //! Density at (R,Z)
  Bool_t mStopped;  //! Scoring has been stopped
  Float_t mRmin;    //! minimum radius
  Float_t mZmax;    //! maximum radius
  Float_t mRmax;    //! maximum z
};
} // namespace steer
} // namespace o2

#endif
