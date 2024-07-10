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

#ifndef ALICEO2_EVENTGEN_FLOWMAPPER_H_
#define ALICEO2_EVENTGEN_FLOWMAPPER_H_

#include "TH1D.h"
#include "TH3D.h"
#include "TF1.h"

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

// this class implements a mapper that introduces a synthetic v2
// into an otherwise uniform initial distribution of phis. It can
// be used, for instance, to create artificial flow of realistic
// intensity in PYTHIA simulations.

// The input histograms are to be read from the CCDB and can
// be customized if necessary. Multiple copies of this mapper
// could be used in case different species should have different
// additional flow.

// N.B.: the main advantages of this mapper is that:
//       1) it preserves total number of particles
//       2) it retains a (distorted) event structure from
//          an original event generator (e.g. PYTHIA)

class FlowMapper
{
 public:
  // Constructor
  FlowMapper();
  
  void Setv2VsPt(TH1D hv2VsPtProvided);
  void SetEccVsB(TH1D hEccVsBProvided);
  
  void CreateLUT(); //to be called if all is set

  Double_t MapPhi(Double_t lPhiInput, TH3D* hLUT, Double_t b, Double_t pt);

  long binsPhi;             // number of phi bins to use
  double precision = 1e-6;  // could be studied
  double derivative = 1e-4; // could be studied
  
  std::unique_ptr<TH1D> hv2vsPt; // input v2 vs pT from measurement
  std::unique_ptr<TH1D> hEccVsB; // ecc vs B (from Glauber MC or elsewhere)
  
  // Cumulative function to be inverted
  std::unique_ptr<TF1> fCumulative;
  
  // the look-up table
  std::unique_ptr<TH3D> hLUT;

  ClassDef(FlowMapper, 1);
};

/*****************************************************************/
/*****************************************************************/

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_FLOWMAPPER_H_ */
