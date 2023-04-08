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

/// \file TopologyDictionary.cxx

#include "ITS3Reconstruction/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "CommonUtils/StringUtils.h"
#include <TFile.h>
#include <iostream>

using std::cout;
using std::endl;
using std::unordered_map;
using std::vector;

ClassImp(o2::its3::TopologyDictionary);
namespace o2
{

namespace its3
{

TopologyDictionary::TopologyDictionary()
{
  memset(mSmallTopologiesLUT, -1, STopoSize * sizeof(int));
}

TopologyDictionary::TopologyDictionary(const std::string& fileName)
{
  readFromFile(fileName);
}

std::ostream& operator<<(std::ostream& os, const its3::TopologyDictionary& dict)
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

void TopologyDictionary::writeBinaryFile(const std::string& outputfile)
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
    file_output.write(const_cast<char*>(reinterpret_cast<const char*>(&p.mPattern.getPattern())),
                      sizeof(unsigned char) * (itsmft::ClusterPattern::kExtendedPatternBytes));
  }
  file_output.close();
}

int TopologyDictionary::readFromFile(const std::string& fname)
{
  if (o2::utils::Str::endsWith(fname, ".root")) {
    std::unique_ptr<TopologyDictionary> d{loadFrom(fname)};
    *this = *d;
  } else if (o2::utils::Str::endsWith(fname, ".bin")) {
    readBinaryFile(fname);
  } else {
    throw std::runtime_error(fmt::format("Unrecognized format {}", fname));
  }
  return 0;
}

int TopologyDictionary::readBinaryFile(const std::string& fname)
{
  mVectorOfIDs.clear();
  mCommonMap.clear();
  for (auto& p : mSmallTopologiesLUT) {
    p = -1;
  }
  std::ifstream in(fname.data(), std::ios::in | std::ios::binary);
  itsmft::GroupStruct gr;
  int groupID = 0;
  if (!in.is_open()) {
    LOG(error) << "The file " << fname << " coud not be opened";
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
      in.read(const_cast<char*>(reinterpret_cast<const char*>(&gr.mPattern.getPattern())), sizeof(unsigned char) * (itsmft::ClusterPattern::kExtendedPatternBytes));
      mVectorOfIDs.push_back(gr);
      if (!gr.mIsGroup) {
        mCommonMap.insert(std::make_pair(gr.mHash, groupID));
        if (gr.mPattern.getUsedBytes() == 1) {
          mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.getByte(2)] = groupID;
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

void TopologyDictionary::getTopologyDistribution(const its3::TopologyDictionary& dict, TH1F*& histo, const char* histName)
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

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(const its3::CompClusterExt& cl) const
{
  LOGP(debug, "Getting cluster coordinates from TopologyDictionaryITS3");
  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers
  math_utils::Point3D<float> locCl;
  if (cl.getSensorID() >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
    locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()) * itsmft::SegmentationAlpide::PitchRow);
    locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()) * itsmft::SegmentationAlpide::PitchCol);
  } else {
    segmentations[cl.getSensorID()].detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
    locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()) * segmentations[cl.getSensorID()].mPitchRow);
    locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()) * segmentations[cl.getSensorID()].mPitchCol);
    float xCurved{0.f}, yCurved{0.f};
    segmentations[cl.getSensorID()].flatToCurved(locCl.X(), locCl.Y(), xCurved, yCurved);
    locCl.SetXYZ(xCurved, yCurved, locCl.Z());
  }
  return locCl;
}

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(const its3::CompClusterExt& cl, const itsmft::ClusterPattern& patt, bool isGroup)
{
  LOGP(debug, "Getting cluster coordinates from TopologyDictionaryITS3");
  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers

  auto refRow = cl.getRow();
  auto refCol = cl.getCol();
  float xCOG = 0, zCOG = 0;
  patt.getCOG(xCOG, zCOG);
  if (isGroup) {
    refRow -= round(xCOG);
    refCol -= round(zCOG);
  }
  math_utils::Point3D<float> locCl;
  if (cl.getSensorID() >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  } else {
    segmentations[cl.getSensorID()].detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
    float xCurved{0.f}, yCurved{0.f};
    segmentations[cl.getSensorID()].flatToCurved(locCl.X(), locCl.Y(), xCurved, yCurved);
    locCl.SetXYZ(xCurved, yCurved, locCl.Z());
  }
  return locCl;
}

template <typename T>
std::array<T, 3> TopologyDictionary::getClusterCoordinatesA(const its3::CompClusterExt& cl) const
{
  LOGP(debug, "Getting cluster coordinates from TopologyDictionaryITS3");
  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers
  std::array<T, 3> locCl;
  if (cl.getSensorID() >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
    locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()) * itsmft::SegmentationAlpide::PitchRow);
    locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()) * itsmft::SegmentationAlpide::PitchCol);
  } else {
    segmentations[cl.getSensorID()].detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
    locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()) * segmentations[cl.getSensorID()].mPitchRow);
    locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()) * segmentations[cl.getSensorID()].mPitchCol);
    float xCurved{0.f}, yCurved{0.f};
    segmentations[cl.getSensorID()].flatToCurved(locCl.X(), locCl.Y(), xCurved, yCurved);
    locCl.SetXYZ(xCurved, yCurved, locCl.Z());
  }
  return locCl;
}

template <typename T>
std::array<T, 3> TopologyDictionary::getClusterCoordinatesA(const its3::CompClusterExt& cl, const itsmft::ClusterPattern& patt, bool isGroup)
{
  LOGP(debug, "Getting cluster coordinates from TopologyDictionaryITS3");
  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers

  auto refRow = cl.getRow();
  auto refCol = cl.getCol();
  float xCOG = 0, zCOG = 0;
  patt.getCOG(xCOG, zCOG);
  if (isGroup) {
    refRow -= round(xCOG);
    refCol -= round(zCOG);
  }
  std::array<T, 3> locCl;
  if (cl.getSensorID() >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  } else {
    segmentations[cl.getSensorID()].detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
    float xCurved{0.f}, yCurved{0.f};
    segmentations[cl.getSensorID()].flatToCurved(locCl.X(), locCl.Y(), xCurved, yCurved);
    locCl.SetXYZ(xCurved, yCurved, locCl.Z());
  }
  return locCl;
}

TopologyDictionary* TopologyDictionary::loadFrom(const std::string& fname, const std::string& objName)
{
  // load object from file
  TFile fl(fname.c_str());
  if (fl.IsZombie()) {
    throw std::runtime_error(fmt::format("Failed to open {} file", fname));
  }
  auto dict = reinterpret_cast<its3::TopologyDictionary*>(fl.GetObjectChecked(objName.c_str(), its3::TopologyDictionary::Class()));
  if (!dict) {
    throw std::runtime_error(fmt::format("Failed to load {} from {}", objName, fname));
  }
  return dict;
}
} // namespace its3
} // namespace o2
