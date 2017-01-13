/// \file Constants.h
/// \brief Definition of constants that should go to a (yet to be defined) parameter space

#ifndef AliceO2_TPC_Constants_H
#define AliceO2_TPC_Constants_H

namespace AliceO2 {
  namespace TPC {

    // gas parameters
    const Float_t WION = 37.3e-6;
    const Float_t ATTCOEF = 250.;
    const Float_t OXYCONT = 5.e-6;
    const Float_t DRIFTV = 2.58;
    const Float_t SIGMAOVERMU = 0.78;
    const Float_t DIFFT = 0.0209;
    const Float_t DIFFL = 0.0221;

    // ROC parameters
    const Float_t EFFGAINGEM1 = 9.1;
    const Float_t EFFGAINGEM2 = 0.88;
    const Float_t EFFGAINGEM3 = 1.66;
    const Float_t EFFGAINGEM4 = 144;
    const Float_t CPAD = 0.1; // in pF

    //electronics parameters
    const Float_t ADCSAT = 1023;
    const Float_t QEL = 1.602e-19;
    const Float_t CHIPGAIN = 20;
    const Float_t ADCDYNRANGE = 2000;     
    const Float_t PEAKINGTIME = 160e-3; // all times are in us

    const Float_t ZBINWIDTH = 0.19379844961;

  }
}

#endif