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

ClassImp(o2::itsmft::TopologyDictionary)

  namespace o2
{
  namespace itsmft
  {

  TopologyDictionary::TopologyDictionary() : mSmallTopologiesLUT{ -1 } {}

  std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dict)
  {
    for (auto& p : dict.mVectorOfGroupIDs) {
      os << "Hash: " << p.mHash << " ErrX: " << p.mErrX << " ErrZ : " << p.mErrZ << " xCOG: " << p.mXCOG << " zCOG: " << p.mZCOG << " Npixles: " << p.mNpixels << " Frequency: "
         << p.mFrequency << std::endl
         << p.mPattern << std::endl
         << "*********************************************************" << std::endl
         << std::endl;
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
      file_output.write(reinterpret_cast<char*>(&p.mPattern.mBitmap),
                        sizeof(unsigned char) * (ClusterPattern::kExtendedPatternBytes));
    }
    file_output.close();
  }

  int TopologyDictionary::ReadBinaryFile(string fname)
  {
    mVectorOfGroupIDs.clear();
    mFinalMap.clear();
    for (auto& p : mSmallTopologiesLUT)
      p = -1;
    std::ifstream in(fname.data(), std::ios::in | std::ios::binary);
    GroupStruct gr;
    int groupID = 0;
    if (!in.is_open()) {
      LOG(ERROR) << "The file coud not be opened";
      throw std::runtime_error("No fired pixels in small topology");
    } else {
      while (in.read(reinterpret_cast<char*>(&gr.mHash), sizeof(unsigned long))) {
        in.read(reinterpret_cast<char*>(&gr.mErrX), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mErrZ), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mXCOG), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mZCOG), sizeof(float));
        in.read(reinterpret_cast<char*>(&gr.mNpixels), sizeof(int));
        in.read(reinterpret_cast<char*>(&gr.mFrequency), sizeof(double));
        in.read(reinterpret_cast<char*>(&gr.mPattern.mBitmap), sizeof(unsigned char) * (ClusterPattern::kExtendedPatternBytes));
        mVectorOfGroupIDs.push_back(gr);
        if (gr.mPattern.getUsedBytes() == 1)
          mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.mBitmap[2]] = groupID;
        if (((gr.mHash) & 0xffffffff) != 0)
          mFinalMap.insert(std::make_pair(gr.mHash, groupID));
        groupID++;
      }
    }
    in.close();
    return 0;
  }

  float TopologyDictionary::GetXcog(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mXCOG;
  }
  float TopologyDictionary::GetErrX(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mErrX;
  }

  float TopologyDictionary::GetZcog(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mZCOG;
  }

  float TopologyDictionary::GetErrZ(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mErrZ;
  }

  unsigned long TopologyDictionary::GetHash(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mHash;
  }

  int TopologyDictionary::GetNpixels(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mNpixels;
  }

  ClusterPattern TopologyDictionary::GetPattern(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else
      return mVectorOfGroupIDs[n].mPattern;
  }

  double TopologyDictionary::GetFrequency(int n)
  {
    if (n < 0 || n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      throw std::range_error("Index out of bounds");
    } else if (n == 0) {
      return mVectorOfGroupIDs[n].mFrequency;
    } else {
      return mVectorOfGroupIDs[n].mFrequency - mVectorOfGroupIDs[n - 1].mFrequency;
    }
  }
  } // namespace itsmft
}
