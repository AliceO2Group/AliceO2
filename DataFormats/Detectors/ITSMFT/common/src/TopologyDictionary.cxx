// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TopologyDictionary.cxx
/// \brief Implementation of the TopologyDictionary class.
///
/// \author Luca Barioglio, University and INFN of Torino

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include <iostream>
#include "ITSMFTBase/SegmentationAlpide.h"

using std::cout;
using std::endl;
using std::string;
using std::unordered_map;
using std::vector;

ClassImp(o2::itsmft::TopologyDictionary);

namespace o2
{
namespace itsmft
{

TopologyDictionary::TopologyDictionary() : mSmallTopologiesLUT{-1} {}

TopologyDictionary::TopologyDictionary(std::string fileName)
{
  readBinaryFile(fileName);
}

std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dict)
{
  int ID = 0;
  for (auto& p : dict.mVectorOfIDs) {
    os << "ID: " << ID++ << " Hash: " << p.mHash << " ErrX: " << p.mErrX << " ErrZ : " << p.mErrZ << " xCOG: " << p.mXCOG << " zCOG: " << p.mZCOG << " Npixles: " << p.mNpixels << " Frequency: " << p.mFrequency << " isGroup : " << std::boolalpha << p.mIsGroup << std::endl
       << p.mPattern << std::endl
       << "*********************************************************" << std::endl
       << std::endl;
  }
  return os;
}

void TopologyDictionary::writeBinaryFile(string outputfile)
{
  std::ofstream file_output(outputfile, std::ios::out | std::ios::binary);
  for (auto& p : mVectorOfIDs) {
    file_output.write(reinterpret_cast<char*>(&p.mHash), sizeof(unsigned long));
    file_output.write(reinterpret_cast<char*>(&p.mErrX), sizeof(float));
    file_output.write(reinterpret_cast<char*>(&p.mErrZ), sizeof(float));
    file_output.write(reinterpret_cast<char*>(&p.mErr2X), sizeof(float));
    file_output.write(reinterpret_cast<char*>(&p.mErr2Z), sizeof(float));
    file_output.write(reinterpret_cast<char*>(&p.mXCOG), sizeof(float));
    file_output.write(reinterpret_cast<char*>(&p.mZCOG), sizeof(float));
    file_output.write(reinterpret_cast<char*>(&p.mNpixels), sizeof(int));
    file_output.write(reinterpret_cast<char*>(&p.mFrequency), sizeof(double));
    file_output.write(reinterpret_cast<char*>(&p.mIsGroup), sizeof(bool));
    file_output.write(reinterpret_cast<char*>(&p.mPattern.mBitmap),
                      sizeof(unsigned char) * (ClusterPattern::kExtendedPatternBytes));
  }
  file_output.close();
}

int TopologyDictionary::readBinaryFile(string fname)
{
  mVectorOfIDs.clear();
  mCommonMap.clear();
  for (auto& p : mSmallTopologiesLUT) {
    p = -1;
  }
  std::ifstream in(fname.data(), std::ios::in | std::ios::binary);
  GroupStruct gr;
  int groupID = 0;
  if (!in.is_open()) {
    LOG(ERROR) << "The file " << fname << " coud not be opened";
    throw std::runtime_error("The file coud not be opened");
  } else {
    while (in.read(reinterpret_cast<char*>(&gr.mHash), sizeof(unsigned long))) {
      in.read(reinterpret_cast<char*>(&gr.mErrX), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.mErrZ), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.mErr2X), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.mErr2Z), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.mXCOG), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.mZCOG), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.mNpixels), sizeof(int));
      in.read(reinterpret_cast<char*>(&gr.mFrequency), sizeof(double));
      in.read(reinterpret_cast<char*>(&gr.mIsGroup), sizeof(bool));
      in.read(reinterpret_cast<char*>(&gr.mPattern.mBitmap), sizeof(unsigned char) * (ClusterPattern::kExtendedPatternBytes));
      mVectorOfIDs.push_back(gr);
      if (!gr.mIsGroup) {
        mCommonMap.insert(std::make_pair(gr.mHash, groupID));
        if (gr.mPattern.getUsedBytes() == 1) {
          mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.mBitmap[2]] = groupID;
        }
      } else {
        mGroupMap.insert(std::make_pair((int)(gr.mHash >> 32) & 0x00000000ffffffff, groupID));
      }
      groupID++;
    }
  }
  in.close();
  return 0;
}

void TopologyDictionary::getTopologyDistribution(const TopologyDictionary& dict, TH1F*& histo, const char* histName)
{
  int dictSize = (int)dict.getSize();
  if (histo) {
    delete histo;
  }
  histo = new TH1F(histName, ";Topology ID;Frequency", dictSize, -0.5, dictSize - 0.5);
  histo->SetFillColor(kRed);
  histo->SetFillStyle(3005);
  histo->SetDrawOption("histo");
  for (int i = 0; i < dictSize; i++) {
    histo->Fill(i, dict.getFrequency(i));
  }
}

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(const CompCluster& cl) const
{
  math_utils::Point3D<float> locCl;
  o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
  locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()));
  locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()));
  return locCl;
}

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(const CompCluster& cl, const ClusterPattern& patt, bool isGroup)
{
  auto refRow = cl.getRow();
  auto refCol = cl.getCol();
  float xCOG = 0, zCOG = 0;
  patt.getCOG(xCOG, zCOG);
  if (isGroup) {
    refRow -= round(xCOG);
    refCol -= round(zCOG);
  }
  math_utils::Point3D<float> locCl;
  o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  return locCl;
}

} // namespace itsmft
} // namespace o2
