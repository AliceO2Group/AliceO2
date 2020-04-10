// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iostream>

#include "TBDigitsFileReader.h"
#include "MCHMappingInterface/Segmentation.h"

using namespace o2::mch;
using namespace std;

TBDigitsFileReader::TBDigitsFileReader()
{
}

void TBDigitsFileReader::init(std::string inputFileName)
{
  mInputFile.open(inputFileName, ios::binary);
  if (!mInputFile.is_open()) {
    throw invalid_argument("Cannot open input file " + inputFileName);
  }
}

bool TBDigitsFileReader::readDigitsFromFile()
{
  int manu2ds[64] = {62, 61, 63, 60, 59, 55, 58, 57, 56, 54, 50, 46, 42, 39, 37, 41,
                     35, 36, 33, 34, 32, 38, 43, 40, 45, 44, 47, 48, 49, 52, 51, 53,
                     7, 6, 5, 4, 2, 3, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23,
                     31, 30, 29, 28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8};

  int ds2manu[64];
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      if (manu2ds[j] != i)
        continue;
      ds2manu[i] = j;
      break;
    }
  }

  digits.clear();
  clusters.clear();

  /// send the digits of the current event

  int event, sievent;
  mInputFile.read(reinterpret_cast<char*>(&event), sizeof(int));
  mInputFile.read(reinterpret_cast<char*>(&sievent), sizeof(int));

  int nDE = 0;
  int DE1 = 0;
  mInputFile.read(reinterpret_cast<char*>(&nDE), sizeof(int));
  if (mInputFile.eof())
    return false;

  for (int iDE = 0; iDE < nDE; iDE++) {

    int DE;
    mInputFile.read(reinterpret_cast<char*>(&DE), sizeof(int));
    float xtrk, ytrk;
    mInputFile.read(reinterpret_cast<char*>(&xtrk), sizeof(float));
    mInputFile.read(reinterpret_cast<char*>(&ytrk), sizeof(float));
    trkx[DE] = xtrk;
    trky[DE] = ytrk;

    int npad;
    mInputFile.read(reinterpret_cast<char*>(&npad), sizeof(int));

    int dsId, dsCh, size, time, padX, padY;
    float charge, posX, posY;
    for (int ih = 0; ih < npad; ih++) {

      mInputFile.read(reinterpret_cast<char*>(&dsId), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&dsCh), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&size), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&time), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&charge), sizeof(float));

      mInputFile.read(reinterpret_cast<char*>(&padX), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&padY), sizeof(int));

      mInputFile.read(reinterpret_cast<char*>(&posX), sizeof(float));
      mInputFile.read(reinterpret_cast<char*>(&posY), sizeof(float));

      uint16_t adc = static_cast<uint16_t>(charge);

      uint16_t detId = DE;
      uint16_t dualSampaId = dsId;
      uint16_t dualSampaChannel = ds2manu[dsCh];

      try {
        mapping::Segmentation segment(detId);
        int padId = segment.findPadByFEE(dualSampaId, dualSampaChannel);
        if (padId < 0) {
          printf("padId: %d\n", padId);
          continue;
        }

        float X = segment.padPositionX(padId);
        float Y = segment.padPositionY(padId);

        if (Y != posY) {
          printf("B  hit %d  dsid=%d chan=%d  charge=%f  time=%d\n",
                 ih, dsId, dsCh, charge, time);
          std::cout << "posY: " << posY << "  " << Y << std::endl;
          break;
        }

        digits.push_back(std::make_unique<Digit>(time, detId, padId, adc));
        //digits.push_back( std::make_unique<Digit>() );
        //Digit* mchdigit = digits.back().get();
        //mchdigit->setDetID(detId);
        //mchdigit->setPadID(padId);
        //mchdigit->setADC(adc);
        //mchdigit->setTimeStamp(time);
      } catch (std::exception& e) {
        continue;
      }
    }

    int nclus;
    mInputFile.read(reinterpret_cast<char*>(&nclus), sizeof(int));

    for (int ic = 0; ic < nclus; ic++) {
      clusters.emplace_back();
      TBCluster& c = clusters.back();
      //mInputFile.read(reinterpret_cast<char*>(&c), sizeof(TBCluster));
      mInputFile.read(reinterpret_cast<char*>(&(c.fDir)), sizeof(char));
      mInputFile.read(reinterpret_cast<char*>(&(c.fNhits)), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&(c.fXNhits)), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&(c.fYNhits)), sizeof(int));
      mInputFile.read(reinterpret_cast<char*>(&(c.fCharge)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fChargemax)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fXclus)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fYclus)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fXmat)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fYmat)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fYmaterror)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fXmaterror)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fChi2mat)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fChmat)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fYcenter)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fXcenter)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fInversepitch)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fK3x)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fK3y)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fK3yrec)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fPadyMean)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fPadxMean)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fPadySigma)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fPadxSigma)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fTimeMean)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fTimeSigma)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fSizeMean)), sizeof(double));
      mInputFile.read(reinterpret_cast<char*>(&(c.fSizeSigma)), sizeof(double));
      //printf("B  clus %d  dir=%d nHits=%d nXHits=%d nYHits=%d Y=%f\n", ic,
      //    (int)c.fDir, (int)c.fNhits, (int)c.fXNhits, (int)c.fYNhits, c.fYmat);
    }
  }

  return true;
}

ssize_t TBDigitsFileReader::getNumberOfDigits()
{
  return digits.size();
}

void TBDigitsFileReader::storeDigits(void* bufferPtr)
{
  Digit* ptr = (Digit*)bufferPtr;
  for (unsigned int di = 0; di < digits.size(); di++) {

    memcpy(ptr, digits[di].get(), sizeof(Digit));
    ptr += 1;
  }
}

std::ostream& operator<<(std::ostream& stream, const TBCluster& c)
{
  stream << "  XNhits: " << c.fXNhits << "\n"
         << "  XYhits: " << c.fYNhits << "\n"
         << "  Charge: " << c.fCharge << "\n"
         << "  fXclus: " << c.fXclus << "\n"
         << "  fYclus: " << c.fYclus << "\n"
         << "  fXmat: " << c.fXmat << "\n"
         << "  fYmat: " << c.fYmat << "\n"
         << "  fXmaterror: " << c.fXmaterror << "\n"
         << "  fYmaterror: " << c.fYmaterror << "\n"
         << "  fChi2mat: " << c.fChi2mat << "\n"
         << "  fK3x: " << c.fK3x << "\n"
         << "  fK3y: " << c.fK3y << "\n"
         << "  fK3yrec: " << c.fK3yrec << "\n";

  return stream;
}
