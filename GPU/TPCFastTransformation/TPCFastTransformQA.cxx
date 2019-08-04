// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransformQA.cxx
/// \brief Implementation of TPCFastTransformQA class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCFastTransformQA.h"
#include "TPCFastTransformManager.h"
#include "TPCFastTransform.h"
#include "AliTPCTransform.h"
#include "AliTPCParam.h"
#include "AliTPCRecoParam.h"
#include "AliTPCcalibDB.h"
#include "AliHLTTPCGeometry.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TStopwatch.h"

#include <iostream>
#include <iomanip>

using namespace GPUCA_NAMESPACE::gpu;
using namespace std;

TPCFastTransformQA::TPCFastTransformQA() {}

int TPCFastTransformQA::doQA(const TPCFastTransform& fastTransform)
{
  const char* fileName = "fastTransformQA.root";

  AliTPCcalibDB* pCalib = AliTPCcalibDB::Instance();
  if (!pCalib) {
    return storeError(-1, "TPCFastTransformQA: No TPC calibration instance found");
  }

  AliTPCParam* tpcParam = pCalib->GetParameters();
  if (!tpcParam) {
    return storeError(-2, "TPCFastTransformQA: No TPCParam object found");
  }

  AliTPCTransform* origTransform = pCalib->GetTransform();
  if (!origTransform) {
    return storeError(-3, "TPCFastTransformQA: No TPC transformation found");
  }

  const AliTPCRecoParam* rec = origTransform->GetCurrentRecoParam();
  if (!rec) {
    return storeError(-5, "TPCFastTransformQA: No TPC Reco Param set in transformation");
  }
  rec->Print();

  int lastTimeBin = rec->GetLastBin();

  // measure execution time
  {
    TStopwatch timer1;
    double nCalls1 = 0;
    double sum1 = 0;
    for (Int_t iSec = 0; iSec < 1; iSec++) {
      cout << "Measure original transformation time for TPC sector " << iSec << " .." << endl;
      int nRows = tpcParam->GetNRow(iSec);
      for (int iRow = 0; iRow < nRows; iRow++) {
        Int_t nPads = tpcParam->GetNPads(iSec, iRow);
        for (float pad = 0.5; pad < nPads; pad += 1.) {
          for (float time = 0; time < lastTimeBin; time++) {
            Int_t is[] = {iSec};
            double orig[3] = {static_cast<Double_t>(iRow), pad, time};
            origTransform->Transform(orig, is, 0, 1);
            nCalls1++;
            sum1 += orig[0] + orig[1] + orig[2];
          }
        }
      }
    }
    timer1.Stop();

    TStopwatch timer2;
    double nCalls2 = 0;
    double sum2 = 0;
    for (Int_t iSec = 0; iSec < 1; iSec++) {
      cout << "Measure fast transformation time for TPC sector " << iSec << " .." << endl;
      int nRows = tpcParam->GetNRow(iSec);
      for (int iRow = 0; iRow < nRows; iRow++) {
        Int_t nPads = tpcParam->GetNPads(iSec, iRow);
        int slice = 0, slicerow = 0;
        AliHLTTPCGeometry::Sector2Slice(slice, slicerow, iSec, iRow);
        for (float pad = 0.5; pad < nPads; pad += 1.) {
          for (float time = 0; time < lastTimeBin; time++) {
            float fast[3];
            fastTransform.Transform(slice, slicerow, pad, time, fast[0], fast[1], fast[2]);
            nCalls2++;
            sum2 += fast[0] + fast[1] + fast[2];
          }
        }
      }
    }
    timer2.Stop();
    cout << "nCalls1 = " << nCalls1 << endl;
    cout << "nCalls2 = " << nCalls2 << endl;
    cout << "Orig transformation    : " << timer1.RealTime() * 1.e9 / nCalls1 << " ns / call" << endl;
    cout << "Fast transformation    : " << timer2.RealTime() * 1.e9 / nCalls2 << " ns / call" << endl;

    cout << "Fast Transformation speedup: " << 1. * timer1.RealTime() / timer2.RealTime() * nCalls2 / nCalls1 << endl;

    int size = sizeof(fastTransform) + fastTransform.getFlatBufferSize();
    cout << "Fast Transformation memory usage: " << size / 1000. / 1000. << " MB" << endl;
    cout << "ignore this " << sum1 << " " << sum2 << endl;
  }

  if (1) {
    TFile* file = new TFile(fileName, "RECREATE");
    if (!file || !file->IsOpen()) {
      return storeError(-1, "Can't recreate QA file !");
    }
    file->cd();
    TNtuple* nt = new TNtuple("fastTransformQA", "fastTransformQA", "sec:row:pad:time:x:y:z:fx:fy:fz");

    for (Int_t iSec = 0; iSec < 1; iSec++) {
      int nRows = tpcParam->GetNRow(iSec);
      for (int iRow = 0; iRow < nRows; iRow++) {
        cout << "Write fastTransform QA for TPC sector " << iSec << ", row " << iRow << " .." << endl;
        Int_t nPads = tpcParam->GetNPads(iSec, iRow);
        int slice = 0, slicerow = 0;
        AliHLTTPCGeometry::Sector2Slice(slice, slicerow, iSec, iRow);
        for (float pad = 0.5; pad < nPads; pad += 1.) {
          for (float time = 0; time < lastTimeBin; time++) {
            Int_t is[] = {iSec};
            double orig[3] = {static_cast<Double_t>(iRow), pad, time};
            float fast[3];
            origTransform->Transform(orig, is, 0, 1);
            fastTransform.Transform(slice, slicerow, pad, time, fast[0], fast[1], fast[2]);
            float entry[] = {(float)iSec, (float)iRow, pad, time, (float)orig[0], (float)orig[1], (float)orig[2], fast[0], fast[1], fast[2]};
            nt->Fill(entry);
          }
        }
      }
    }
    file->Write();
    file->Close();
    delete file;
  }
  return 0;
}

int TPCFastTransformQA::doQA(Long_t TimeStamp)
{
  TPCFastTransform fastTransform;
  TPCFastTransformManager man;

  man.create(fastTransform, nullptr, TimeStamp);

  return doQA(fastTransform);
}
