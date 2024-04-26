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

#include "EMCALCalibration/EMCALPedestalHelper.h"
using namespace o2::emcal;

std::vector<char> EMCALPedestalHelper::createPedestalInstruction(const Pedestal& obj, const int runNum)
{

  setZero();

  o2::emcal::Geometry* geo = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  o2::emcal::MappingHandler mapper;

  for (int itower = 0; itower < 17664; itower++) {
    auto [ddl, row, col] = geo->getOnlineID(itower);
    auto [sm, mod, iphi, ieta] = geo->GetCellIndex(itower);
    const auto& mapping = mapper.getMappingForDDL(ddl);
    int ircu = ddl % 2;
    auto addressLG = mapping.getHardwareAddress(row, col, o2::emcal::ChannelType_t::LOW_GAIN),
         addressHG = mapping.getHardwareAddress(row, col, o2::emcal::ChannelType_t::HIGH_GAIN);
    auto fecLG = o2::emcal::Channel::getFecIndexFromHwAddress(addressLG),
         fecHG = o2::emcal::Channel::getFecIndexFromHwAddress(addressHG),
         branchLG = o2::emcal::Channel::getBranchIndexFromHwAddress(addressLG),
         branchHG = o2::emcal::Channel::getBranchIndexFromHwAddress(addressHG),
         chipLG = o2::emcal::Channel::getAltroIndexFromHwAddress(addressLG),
         chipHG = o2::emcal::Channel::getAltroIndexFromHwAddress(addressHG),
         channelLG = o2::emcal::Channel::getChannelIndexFromHwAddress(addressLG),
         channelHG = o2::emcal::Channel::getChannelIndexFromHwAddress(addressHG);
    fMeanPed[sm][ircu][branchHG][fecHG][chipHG][channelHG] = obj.getPedestalValue(itower, false, false);
    fMeanPed[sm][ircu][branchLG][fecLG][chipLG][channelLG] = obj.getPedestalValue(itower, true, false);
  }

  for (int iledmon = 0; iledmon < 480; iledmon++) {
    int sm = iledmon / 24,
        col = iledmon % 24,
        ircu = 0, // LEDMONS always on RCU 0
      iddl = 2 * sm + ircu;
    const auto& mapping = mapper.getMappingForDDL(iddl);
    auto addressLG = mapping.getHardwareAddress(0, col, o2::emcal::ChannelType_t::LEDMON),
         addressHG = mapping.getHardwareAddress(1, col, o2::emcal::ChannelType_t::LEDMON);
    auto fecLG = o2::emcal::Channel::getFecIndexFromHwAddress(addressLG),
         fecHG = o2::emcal::Channel::getFecIndexFromHwAddress(addressHG),
         branchLG = o2::emcal::Channel::getBranchIndexFromHwAddress(addressLG),
         branchHG = o2::emcal::Channel::getBranchIndexFromHwAddress(addressHG),
         chipLG = o2::emcal::Channel::getAltroIndexFromHwAddress(addressLG),
         chipHG = o2::emcal::Channel::getAltroIndexFromHwAddress(addressHG),
         channelLG = o2::emcal::Channel::getChannelIndexFromHwAddress(addressLG),
         channelHG = o2::emcal::Channel::getChannelIndexFromHwAddress(addressHG);
    fMeanPed[sm][ircu][branchHG][fecHG][chipHG][channelHG] = obj.getPedestalValue(iledmon, false, true);
    fMeanPed[sm][ircu][branchLG][fecLG][chipLG][channelLG] = obj.getPedestalValue(iledmon, true, true);
  }

  return createInstructionString(runNum);
}

void EMCALPedestalHelper::setZero()
{
  for (int ism = 0; ism < kNSM; ism++) {
    for (int ircu = 0; ircu < kNRCU; ircu++) {
      for (int ibranch = 0; ibranch < kNBranch; ibranch++) {
        for (int ifec = 0; ifec < kNFEC; ifec++) {
          for (int ichip = 0; ichip < kNChip; ichip++) {
            for (int ichan = 0; ichan < kNChan; ichan++) {
              fMeanPed[ism][ircu][ibranch][ifec][ichip][ichan] = 0;
            }
          }
        }
      }
    }
  }
}

std::vector<char> EMCALPedestalHelper::createInstructionString(const int runNum)
{
  std::stringstream fout;

  if (runNum > 0) {
    fout << runNum << std::endl;
  }

  unsigned int lineValue = 0;

  const unsigned int FECheaderCode = 0xC0000000;
  //  const unsigned int FECwordCode   = 0x80000000;
  const unsigned int FEClineCode = 0x40000000;

  const unsigned int TrailerLineCode = 0xFFFFFFFF;

  short iSM = 0;
  short iRCU = 0;
  short ibranch = 0;
  short iFEC = 0;
  short ichip = 0;
  short ichan = 0;
  short Ped = 0;
  short iDTC = 0;

  for (iSM = 0; iSM < kNSM; iSM++) {
    int iside = iSM % 2;
    int isect = iSM / 2;
    if (iSM > 11) {
      isect += 3; // skip non-installed sectors
    }

    std::bitset<kNDTC> activeDTC;
    for (iDTC = 0; iDTC < kNDTC; iDTC++) {
      if (iDTC == 10 || iDTC == 20 || iDTC == 30) { // skip TRU
        activeDTC[iDTC] = 0;
      } else {
        if (iSM < 10) { // not special third SMs or DCal SMs
          activeDTC[iDTC] = 1;
        } else {
          if (iSM == 10 || iSM == 19) { // SMA5 or SMC12
            if (iDTC < 14) {
              activeDTC[iDTC] = 1;
            } else {
              activeDTC[iDTC] = 0;
            }
          } else if (iSM == 11 || iSM == 18) { // SMC5 or SMA12
            if (iDTC == 0 || iDTC >= 27) {
              activeDTC[iDTC] = 1;
            } else {
              activeDTC[iDTC] = 0;
            }
          } else {
            // DCal... no FECs in  9,11-13, 23-26, 36-39
            if ((iDTC >= 9 && iDTC <= 13) || (iDTC >= 23 && iDTC <= 26) ||
                (iDTC >= 36 && iDTC <= 39)) {
              activeDTC[iDTC] = 0;
            } else {
              activeDTC[iDTC] = 1;
            }
          } // DCal
        }   // non-EMCal
      }     // non-TRU
    }

    // OK, let's generate the files for all active FECs/DTCs
    for (iDTC = 0; iDTC < kNDTC; iDTC++) {
      if (activeDTC[iDTC] == 0) {
        continue;
      }

      lineValue = FECheaderCode | isect << 9 | iside << 8 | iDTC;
      fout << lineValue << std::endl;

      iRCU = iDTC / 20;
      ibranch = (iDTC % 20) / 10;
      iFEC = iDTC % 10;
      int ipos = iFEC + 10 * ibranch;

      int dtcselUpper = 0;
      int dtcselLower = 0;
      if (iRCU == 0) {
        dtcselLower = (1 << ipos);
      } else { // crate == 1
        dtcselUpper = (1 << ipos);
      }

      for (ichip = 0; ichip < kNChip; ichip++) { // ALTRO 0,2,3,4
        if (ichip != 1) {
          for (ichan = 0; ichan < kNChan; ichan++) {
            if (iFEC != 0 || (ichan < 8 || ichan > 11)) {
              Ped = fMeanPed[iSM][iRCU][ibranch][iFEC][ichip][ichan];
              int writeAddr = (ichip << 4) | ichan;
              lineValue = FEClineCode | (writeAddr << 12) | Ped;
              fout << lineValue << std::endl;
            }
          }
        }
      } // chip

    } // iDTC
  }   // iSM

  if (runNum > 0) {
    fout << TrailerLineCode << std::endl;
  }

  const std::string instructionString(fout.str());
  std::vector<char> output(instructionString.begin(), instructionString.end());
  return output;
}

void EMCALPedestalHelper::dumpInstructions(const std::string_view filename, const gsl::span<char>& data)
{
  std::ofstream fout(filename.data());
  fout << data.data();
  fout.close();
}