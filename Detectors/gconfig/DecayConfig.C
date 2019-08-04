/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TPythia6Decayer.h"
#include "TPythia6.h"
#include "TVirtualMC.h"
#endif

void DecayConfig()
{

  // This script uses the external decayer TPythia6Decayer in place of the
  // concrete Monte Carlo native decay mechanisms only for the
  // specific types of decays defined below.

  // Access the external decayer singleton and initialize it
  TPythia6Decayer* decayer = TPythia6Decayer::Instance();
  // The following just tells pythia6 to not decay particles only to
  // certain channels.

  decayer->SetForceDecay(TPythia6Decayer::kAll);
  //example:  Force the J/PSI decay channel e+e-
  //        Int_t products[2];
  //        Int_t mult[2];
  //        Int_t npart=2;

  //decay products
  //        products[0]=11;
  //        products[1]=-11;
  //multiplicity
  //        mult[0]=1;
  //        mult[1]=1;
  // force the decay channel
  //        decayer->ForceParticleDecay(443,products,mult,npart);

  decayer->Init();

  // Tell the concrete monte carlo to use the external decayer.  The
  // external decayer will be used for:
  // i)particle decays not defined in concrete monte carlo, or
  //ii)particles for which the concrete monte carlo is told
  //   to use the external decayer for its type via:
  //     TVirtualMC::GetMC()->SetUserDecay(pdgId);
  //   If this is invoked, the external decayer will be used for particles
  //   of type pdgId even if the concrete monte carlo has a decay mode
  //   already defined for that particle type.
  TVirtualMC::GetMC()->SetExternalDecayer(decayer);

  TPythia6& pythia6 = *(TPythia6::Instance());

  // The pythia6 decayer is used in place of the concrete Monte Carlo
  // decay for the particles type mu+/-,pi+/-, K+/-, K0L in order to preserve
  // the decay product neutrino flavor, which is otherwise not preserved in
  // Geant3 decays.
  const Int_t npartnf = 9;
  // mu-,mu+,pi+,pi-,K+,K-,K0L, Xi-
  Int_t pdgnf[npartnf] = {13, -13, 211, -211, 321, -321, 130, 3312, 443};
  for (Int_t ipartnf = 0; ipartnf < npartnf; ipartnf++) {
    Int_t ipdg = pdgnf[ipartnf];

    if (TString(TVirtualMC::GetMC()->GetName()) == "TGeant3")
      TVirtualMC::GetMC()->SetUserDecay(ipdg); // Force the decay to be done w/external decayer

    pythia6.SetMDCY(pythia6.Pycomp(ipdg), 1, 1); // Activate decay in pythia
  }

  // The following will print the decay modes
  pythia6.Pyupda(1, 6);

  // rho0 (113), rho+ (213), rho- (-213) and
  // D+(411) ,D-(-411),D0(421),D0bar(-421) have decay modes defined in
  // TGeant3::DefineParticles, but for these particles
  // those decay modes are overridden to make use of pythia6.
  const Int_t nparthq = 3;
  // rho0,rho+,rho-,D+,D-,D0,D0bar
  //Int_t pdghq[nparthq] = {113,213,-213,411,-411,421,-421};
  Int_t pdghq[nparthq] = {421, 3122, -3122};
  for (Int_t iparthq = 0; iparthq < nparthq; iparthq++) {
    Int_t ipdg = pdghq[iparthq];
    if (TString(TVirtualMC::GetMC()->GetName()) == "TGeant3")
      TVirtualMC::GetMC()->SetUserDecay(ipdg);   // Force the decay to be done w/external decayer
    pythia6.SetMDCY(pythia6.Pycomp(ipdg), 1, 1); // Activate decay in pythia
  }
  // Set pi0 to be stable in pythia6 so that Geant3 can handle decay.
  // In general, TGeant3 is set up through TGeant3gu::gudcay to pass
  // all pythia6 decay products back to the G3 transport mechanism if they
  // have a lifetime > 1.E-15 sec for further transport.
  // Since the pi0 lifetime is less than this, if pi0 is produced as a decay
  // product in pythia6, e.g. KL0 -> pi0 pi+ pi-, the pi0 will be immediately
  // decayed by pythia6 to 2 gammas, and the KL0 decay product list passed
  // back to the transport mechanism will be "gamma gamma pi+ pi-", i.e.
  // the pi0 will not be visible in the list of secondaries passed back to
  // the transport mechanism and will not be pushed to the stack for possible
  // storage to the stdhep output array.
  // To avoid this, the pi0 is set to stable in pythia6, and its decay
  // will be handled by Geant3.
  //pythia6.SetMDCY(pythia6.Pycomp(111),1,0);
  //}
}
