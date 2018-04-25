#ifndef ALIHLTTPCREVERSETRANSFORMINFOV1DATA_H
#define ALIHLTTPCREVERSETRANSFORMINFOV1DATA_H

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

/** @author David Rohr
*/

struct AliHLTTPCReverseTransformInfoV1
{
  AliHLTTPCReverseTransformInfoV1() : fNTBinsL1(0.f), fZWidth(0.f), fZSigma(0.f), fZLengthA(0.f), fZLengthC(0.f), fDriftCorr(0.f),
    fTime0corrTimeA(0.f), fDeltaZcorrTimeA(0.f), fVdcorrectionTimeA(0.f), fVdcorrectionTimeGYA(0.f),
    fTime0corrTimeC(0.f), fDeltaZcorrTimeC(0.f), fVdcorrectionTimeC(0.f), fVdcorrectionTimeGYC(0.f),
    fDriftTimeFactorA(0.f), fDriftTimeOffsetA(0.f), fDriftTimeFactorC(0.f), fDriftTimeOffsetC(0.f),
    fCorrectY1(0.f), fCorrectY2(0.f), fCorrectY3(0.f)
    {}
  float fNTBinsL1;
  float fZWidth;
  float fZSigma;
  float fZLengthA;
  float fZLengthC;
  float fDriftCorr;
  float fTime0corrTimeA;
  float fDeltaZcorrTimeA;
  float fVdcorrectionTimeA;
  float fVdcorrectionTimeGYA;
  float fTime0corrTimeC;
  float fDeltaZcorrTimeC;
  float fVdcorrectionTimeC;
  float fVdcorrectionTimeGYC;

  float fDriftTimeFactorA;
  float fDriftTimeOffsetA;
  float fDriftTimeFactorC;
  float fDriftTimeOffsetC;
  
  float fCorrectY1;
  float fCorrectY2;
  float fCorrectY3;
};

#endif
