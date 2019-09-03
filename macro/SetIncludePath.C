#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <cstdio>
#include <iostream>
#include "TString.h"
#include "TSystem.h"
#endif

void SetIncludePath()
{
  TString dir = getenv("VMCWORKDIR");
  if (!dir.IsWhitespace()) {
    char inc1[100];
    sprintf(inc1, " -I%s/Detectors/ITSMFT/ITS/base/include/", dir.Data());
    char inc2[100];
    sprintf(inc2, " -I%s/Detectors/Passive/include/", dir.Data());
    char inc3[100];
    sprintf(inc3, " -I%s/Detectors/ITSMFT/ITS/simulation/include/", dir.Data());
    char inc4[100];
    sprintf(inc4, " -I%s/Detectors/TPC/simulation/include/ ", dir.Data());
    char inc5[100];
    sprintf(inc5, " -I%s/Detectors/TPC/simulation/", dir.Data());
    char inc6[100];
    sprintf(inc6, " -I%s/Detectors/Base/include/", dir.Data());
    char inc7[100];
    sprintf(inc7, " -I%s/Common/Field/include/", dir.Data());
    char inc8[100];
    sprintf(inc8, " -I%s/Common/MathUtils/include/", dir.Data());
    char inc9[100];
    sprintf(inc9, " -I%s/Detectors/ITSMFT/ITS/reconstruction/include/", dir.Data());
    char inc10[100];
    sprintf(inc10, " -I%s/Detectors/ITSMFT/common/base/include/", dir.Data());
    char inc11[100];
    sprintf(inc11, " -I%s/Detectors/ITSMFT/common/simulation/include/", dir.Data());
    char inc12[100];
    sprintf(inc12, " -I%s/Detectors/ITSMFT/common/reconstruction/include/", dir.Data());
    char inc13[100];
    sprintf(inc13, " -I%s/Detectors/ITSMFT/MFT/base/include/", dir.Data());
    char inc14[100];
    sprintf(inc14, " -I%s/Detectors/ITSMFT/MFT/simulation/include/", dir.Data());
    char inc15[100];
    sprintf(inc15, " -I%s/Detectors/ITSMFT/MFT/reconstruction/include/", dir.Data());

    TString includePath = inc1;
    includePath += inc2;
    includePath += inc3;
    includePath += inc4;
    includePath += inc5;
    includePath += inc6;
    includePath += inc7;
    includePath += inc8;
    includePath += inc9;
    includePath += inc10;
    includePath += inc11;
    includePath += inc12;
    includePath += inc13;
    includePath += inc14;
    includePath += inc15;

    gSystem->AddIncludePath(includePath.Data());
    cout << "Added " << endl
         << includePath.Data() << endl
         << " to include paths" << endl;
  } else {
    cout << endl
         << endl;
    cout << "VMCWORKDIR is not defined, please source config.sh(.csh)." << endl;
  }
}
