/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/


//-------------------------------------------------------------------------
//            Helper -DATA  class ESD V0 vertex class
//            contains effective errror parameterization 
//            Effective parameterization of resolution in DCA and PA  as function of radii and momenta
//            Mini-max coeficient    fPMinFraction... fPMaxFraction...
//                                   as limits for Error parameterization using Covariance matrix             
//            For detailes : see  AliESDv0 class   
//             
//    Origin: Marian Ivanov marian.ivanov@cern.ch
//-------------------------------------------------------------------------


#include "AliESDV0Params.h"


ClassImp(AliESDV0Params)




AliESDV0Params::AliESDV0Params() :
  TObject(),
  // These constants are used in the error parameterization using covariance matrix
  // minimal sigma in AP and DCA 
  fPSigmaOffsetD0(0.03),      // minimal sigma of error estimate
  fPSigmaOffsetAP0(0.005),
  //
  // Effective parameterization of DCA resolution as function of pt and decay radii
  //
  fPSigmaMaxDE(0.5),
  fPSigmaOffsetDE(0.06),
  fPSigmaCoefDE(0.02),
  fPSigmaRminDE(2.7),
  //
  //
  // Effective parameterization of PA resolution as function of pt and decay radii
  //
  fPSigmaBase0APE(0.005),
  fPSigmaMaxAPE(0.06),
  fPSigmaR0APE(0.02),
  fPSigmaR1APE(0.1), 
  fPSigmaP0APE(0.7*0.4),
  fPSigmaP1APE(0.3*0.4),
  //
  //
  // Minimax parameters
  //
  fPMinFractionAP0(0.5),
  fPMaxFractionAP0(1.5),
  fPMinAP0(0.003),
  fPMinFractionD0(0.5),
  fPMaxFractionD0(1.5),
  fPMinD0(0.05),
  fkMaxDist0(0.1),
  fkMaxDist1(0.1),
  fkMaxDist(1.),
  fkMinPointAngle(0.85),
  fkMinPointAngle2(0.99),
  fkMinR(0.5),
  fkMaxR(220.),
  fkMinPABestConst(0.9999),
  fkMaxRBestConst(10.),
  fkCausality0Cut(0.19),
  fkLikelihood01Cut(0.45),
  fkLikelihood1Cut(0.5),
  fkCombinedCut(0.55),
  fkMinClFullTrk(5.0),
  fkMinTgl0(1.05),
  
  fkMinClForb0(4.5),
  fkMinRTgl0(40.), 
  fkMinNormDistForbTgl0(3.0),
  fkMinNormDistForb1(3.0),
  fkMinNormDistForb2(2.0),
  fkMinNormDistForb3(1.0),
  fkMinNormDistForb4(4.0),
  fkMinNormDistForb5(5.0),
  fkMinNormDistForbProt(2.0),
  fkMaxPidProbPionForb(0.5),
  fkMinRTPCdensity(40.),
  fkMaxRTPCdensity0(110.),
  fkMaxRTPCdensity10(120.),
  fkMaxRTPCdensity20(130.),
  fkMaxRTPCdensity30(140.),
  
  fkMinTPCdensity(0.6),
  fkMinTgl1(1.1),
  fkMinTgl2(1.),
  fkMinchi2before0(16.),
  fkMinchi2before1(16.),
  fkMinchi2after0(16.),
  fkMinchi2after1(16.),
  fkAddchi2SharedCl(18.),
  fkAddchi2NegCl0(25.),
  fkAddchi2NegCl1(30.),
  
  fkSigp0Par0(0.0001),
  fkSigp0Par1(0.001),
  fkSigp0Par2(0.1),
  fkSigpPar0(0.5),
  fkSigpPar1(0.6),
  fkSigpPar2(0.4),
  fkMaxDcaLh0(0.5),
  
  fkChi2KF(100.),
  fkRobustChi2KF(100.),
  fgStreamLevel(0)

  //

{
  //
  // Default constructor
  // Consult AliESDv0 to see actual error parameterization 
  //
}



