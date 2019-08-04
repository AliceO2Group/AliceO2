// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/format.hpp>

#include "TFile.h"
#include "TTree.h"
#include "TMath.h"

#include "MathUtils/MathBase.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"

#include "TPCCalibration/CalibTreeDump.h"

using boost::format;
using boost::str;
using o2::math_utils::math_base::median;

using namespace o2::tpc;

//______________________________________________________________________________
void CalibTreeDump::dumpToFile(const std::string filename)
{
  // ===| open file and crate tree |============================================
  std::unique_ptr<TFile> file(TFile::Open(filename.c_str(), "recreate"));
  auto tree = new TTree("calibTree", "Calibration data object tree");

  // ===| add default mappings |================================================
  addDefaultMapping(tree);

  // ===| fill calDet objects |=================================================
  addCalDetObjects(tree);

  file->Write();
}

//______________________________________________________________________________
void CalibTreeDump::addDefaultMapping(TTree* tree)
{
  // loop over ROCs
  //

  // ===| mapper |==============================================================
  const auto& mapper = Mapper::instance();

  // ===| default mapping objects |=============================================
  uint16_t rocNumber = 0;
  // positions
  std::vector<float> gx;
  std::vector<float> gy;
  std::vector<float> lx;
  std::vector<float> ly;
  // row and pad
  std::vector<unsigned short> row;
  std::vector<unsigned short> pad;
  std::vector<unsigned short> cpad;

  // ===| add branches with default mappings |==================================
  tree->Branch("roc", &rocNumber);
  tree->Branch("gx", &gx);
  tree->Branch("gy", &gy);
  tree->Branch("lx", &lx);
  tree->Branch("ly", &ly);

  // ===| loop over readout chambers |==========================================
  for (ROC roc; !roc.looped(); ++roc) {
    rocNumber = roc;

    // ===| clear position vectors |============================================
    gx.clear();
    gy.clear();
    lx.clear();
    ly.clear();

    // ===| loop over pad rows |================================================
    const int numberOfRows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < numberOfRows; ++irow) {

      // ===| loop over pads in row |===========================================
      const int numberOfPadsInRow = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < numberOfPadsInRow; ++ipad) {
        const PadROCPos padROCPos(rocNumber, irow, ipad);
        const PadPos padPos = mapper.getGlobalPadPos(padROCPos); // pad and row in sector
        const PadCentre& localPadXY = mapper.getPadCentre(padPos);
        const LocalPosition2D globalPadXY = mapper.getPadCentre(padROCPos);

        gx.emplace_back(globalPadXY.X());
        gy.emplace_back(globalPadXY.Y());
        lx.emplace_back(localPadXY.X());
        ly.emplace_back(localPadXY.Y());

        row.emplace_back(irow);
        pad.emplace_back(ipad);
        cpad.emplace_back(ipad - numberOfPadsInRow / 2);
      }
    }

    tree->Fill();
  }
}

//______________________________________________________________________________
void CalibTreeDump::addCalDetObjects(TTree* tree)
{

  // declare forwarding visitors for mean and median
  auto visitorMean = make_forwarding_visitor<double>([](const auto& t) { return TMath::Mean(t.begin(), t.end()); });
  auto visitorMedian = make_forwarding_visitor<double>([](const auto& t) { return TMath::Median(t.size(), t.data()); });

  auto visitorMeanVector = make_forwarding_visitor<std::vector<float>>([](const auto& t) {
    std::vector<float> values;
    // ===| loop over ROCs and fill |===
    for (const auto& calArray : t.getData()) {
      auto& data = calArray.getData();
      values.emplace_back(float(TMath::Mean(data.begin(), data.end())));
    }
    return values;
  });

  auto visitorMedianVector = make_forwarding_visitor<std::vector<float>>([](const auto& t) {
    std::vector<float> values;
    // ===| loop over ROCs and fill |===
    for (const auto& calArray : t.getData()) {
      auto& data = calArray.getData();
      values.emplace_back(float(TMath::Median(data.size(), data.data())));
    }
    return values;
  });

  auto visitorValVector = make_forwarding_visitor<std::vector<std::vector<float>>>([](const auto& t) {
    std::vector<std::vector<float>> values;
    // ===| loop over ROCs and fill |===
    for (const auto& calArray : t.getData()) {
      values.emplace_back();
      auto& vector = values.back();
      auto& data = calArray.getData();
      for (const auto& val : data)
        vector.emplace_back(float(val));
    }
    return values;
  });

  auto visitorName = make_forwarding_visitor<std::string>([](const auto& t) { return t.getName(); });
  //auto visitorData = make_forwarding_visitor<std::vector>([](const auto& t){ return t.getName();});

  int iter = 0;
  for (auto& calDet : mCalDetObjects) {
    // ===| branch names |===
    //std::string name = calDet.getName();
    std::string name = boost::apply_visitor(visitorName, calDet);

    if (name == "PadCalibrationObject" || name.size() == 0) {
      name = str(format("calDet_%1$02d") % iter);
    }

    std::string meanName = str(format("%1%_mean") % name);
    std::string medianName = str(format("%1%_median") % name);

    // ===| branch variables |===
    std::vector<float> value;
    float mean;
    float median;

    // ===| branch definitions |===
    TBranch* brData = tree->Branch(name.data(), &value);
    TBranch* brMean = tree->Branch(meanName.data(), &mean);
    TBranch* brMedian = tree->Branch(medianName.data(), &median);

    auto meanVector = boost::apply_visitor(visitorMeanVector, calDet);
    //auto medianVector = boost::apply_visitor(visitorMedianVector, calDet);
    auto valueVector = boost::apply_visitor(visitorValVector, calDet);

    // ===| loop over ROCs and fill |===
    //for (const auto& calArray : calDet.getData()) {
    //auto& data = calArray.getData();

    //value.clear();

    //// ---| statistics |---
    //mean   = boost::apply_visitor(visitorMean,   data);
    //median = boost::apply_visitor(visitorMedian, data);

    //for (const auto& val : data) value.emplace_back(val);

    //brData->Fill();
    //brMean->Fill();
    //brMedian->Fill();
    //}
  }
}
