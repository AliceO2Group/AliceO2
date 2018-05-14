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
#include <iostream>

using std::cout;
using std::endl;
using std::string;
using std::unordered_map;
using std::vector;

ClassImp(o2::ITSMFT::TopologyDictionary)

  namespace o2
{
  namespace ITSMFT
  {
  std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dict)
  {
    for (auto& p : dict.mVectorOfGroupIDs) {
      os << p.mHash << " " << p.mErrX << " " << p.mErrZ << " " << p.mXCOG << " " << p.mZCOG << " " << p.mNpixels << " "
         << p.mFrequency << std::endl;
    }
    return os;
  }

  void TopologyDictionary::WriteBinaryFile(string outputfile)
  {
    std::ofstream file_output(outputfile, std::ios::out | std::ios::binary);
    for (auto& p : mVectorOfGroupIDs) {
      file_output.write(reinterpret_cast<char*>(&p.mHash), sizeof(unsigned long));
      file_output.write(reinterpret_cast<char*>(&p.mErrX), sizeof(float));
      file_output.write(reinterpret_cast<char*>(&p.mErrZ), sizeof(float));
      file_output.write(reinterpret_cast<char*>(&p.mXCOG), sizeof(float));
      file_output.write(reinterpret_cast<char*>(&p.mZCOG), sizeof(float));
      file_output.write(reinterpret_cast<char*>(&p.mNpixels), sizeof(int));
      file_output.write(reinterpret_cast<char*>(&p.mFrequency), sizeof(double));
    }
    file_output.close();
  }

  void TopologyDictionary::ReadFile(string fname)
  {
    mVectorOfGroupIDs.clear();
    mFinalMap.clear();
    std::ifstream in(fname);
    GroupStruct gr;
    int groupID = 0;
    if (!in.is_open()) {
      cout << "The file could not be opened" << endl;
      exit(1);
    } else {
      while (in >> gr.mHash >> gr.mErrX >> gr.mErrZ >> gr.mXCOG >> gr.mZCOG >> gr.mNpixels >> gr.mFrequency) {
        mVectorOfGroupIDs.push_back(gr);
        if (((gr.mHash) & 0xffffffff) != 0)
          mFinalMap.insert(std::make_pair(gr.mHash, groupID));
        groupID++;
      }
    }
    in.close();
  }

  void TopologyDictionary::ReadBinaryFile(string fname)
  {
    mVectorOfGroupIDs.clear();
    mFinalMap.clear();
    std::ifstream in(fname.data(), std::ios::in | std::ios::binary);
    GroupStruct gr;
    int groupID = 0;
    if (!in.is_open()) {
      cout << "The file could not be opened" << endl;
      exit(1);
    } else {
      while (in.read(reinterpret_cast<char*>(&gr.mHash), sizeof(unsigned long))) {
        in.read(reinterpret_cast<char*>(&gr.mErrX), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mErrZ), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mXCOG), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mZCOG), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mNpixels), sizeof(int));
        in.read(reinterpret_cast<char*>(&gr.mFrequency), sizeof(double));
        mVectorOfGroupIDs.push_back(gr);
        if (((gr.mHash) & 0xffffffff) != 0)
          mFinalMap.insert(std::make_pair(gr.mHash, groupID));
        groupID++;
      }
    }
    in.close();
  }
  } // namespace ITSMFT
}
