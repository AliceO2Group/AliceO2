// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//root includes
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"

//o2 includes
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/ClusterNative.h"

ClassImp(o2::tpc::qc::Clusters);

using namespace o2::tpc::qc;

//______________________________________________________________________________
bool Clusters::processCluster(const o2::tpc::ClusterNative& cluster, const o2::tpc::Sector sector, const int row)
{
  const int nROC = row < 63 ? int(sector) : int(sector) + 36;
  const int rocRow = row < 63 ? row : row - 63;

  const float pad = cluster.getPad();

  const uint16_t qMax = cluster.qMax;
  const uint16_t qTot = cluster.qTot;
  const float sigmaPad = cluster.getSigmaPad();
  const float sigmaTime = cluster.getSigmaTime();
  const float timeBin = cluster.getTime();

  float count = mNClusters.getCalArray(nROC).getValue(rocRow, pad);
  mNClusters.getCalArray(nROC).setValue(rocRow, pad, count + 1);

  float charge = mQMax.getCalArray(nROC).getValue(rocRow, pad);
  mQMax.getCalArray(nROC).setValue(rocRow, pad, charge + qMax);

  charge = mQTot.getCalArray(nROC).getValue(rocRow, pad);
  mQTot.getCalArray(nROC).setValue(rocRow, pad, charge + qTot);

  count = mSigmaTime.getCalArray(nROC).getValue(rocRow, pad);
  mSigmaTime.getCalArray(nROC).setValue(rocRow, pad, count + sigmaTime);

  count = mSigmaPad.getCalArray(nROC).getValue(rocRow, pad);
  mSigmaPad.getCalArray(nROC).setValue(rocRow, pad, count + sigmaPad);

  count = mTimeBin.getCalArray(nROC).getValue(rocRow, pad);
  mTimeBin.getCalArray(nROC).setValue(rocRow, pad, count + timeBin);

  return true;
}

//______________________________________________________________________________
void Clusters::analyse()
{
  mQMax /= mNClusters;
  mQTot /= mNClusters;
  mSigmaTime /= mNClusters;
  mSigmaPad /= mNClusters;
  mTimeBin /= mNClusters;
}

//______________________________________________________________________________
void Clusters::dumpToFile(std::string filename)
{
  if (filename.find(".root") != std::string::npos) {
    filename.resize(filename.size() - 5);
  }

  std::string canvasFile = filename + "_canvas.root";
  auto f = std::unique_ptr<TFile>(TFile::Open(canvasFile.c_str(), "recreate"));
  f->WriteObject(o2::tpc::painter::draw(mNClusters), mNClusters.getName().data());
  f->WriteObject(o2::tpc::painter::draw(mQMax), mQMax.getName().data());
  f->WriteObject(o2::tpc::painter::draw(mQTot), mQTot.getName().data());
  f->WriteObject(o2::tpc::painter::draw(mSigmaTime), mSigmaTime.getName().data());
  f->WriteObject(o2::tpc::painter::draw(mSigmaPad), mSigmaPad.getName().data());
  f->WriteObject(o2::tpc::painter::draw(mTimeBin), mTimeBin.getName().data());
  f->Close();

  std::string calPadFile = filename + ".root";
  auto g = std::unique_ptr<TFile>(TFile::Open(calPadFile.c_str(), "recreate"));
  g->WriteObject(&mNClusters.getData(), mNClusters.getName().data());
  g->WriteObject(&mQMax.getData(), mQMax.getName().data());
  g->WriteObject(&mQTot.getData(), mQTot.getName().data());
  g->WriteObject(&mSigmaTime.getData(), mSigmaTime.getName().data());
  g->WriteObject(&mSigmaPad.getData(), mSigmaPad.getName().data());
  g->WriteObject(&mTimeBin.getData(), mTimeBin.getName().data());
  g->Close();
}