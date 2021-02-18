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
#include <boost/format.hpp>
#include "CommonConstants/LHCConstants.h"
#include "EMCALReconstruction/RCUTrailer.h"

using namespace o2::emcal;

void RCUTrailer::reset()
{
  mRCUId = -1;
  mFirmwareVersion = 0;
  mTrailerSize = 0;
  mPayloadSize = 0;
  mFECERRA = 0;
  mFECERRB = 0;
  mERRREG2 = 0;
  mERRREG3 = 0;
  mActiveFECsA = 0;
  mActiveFECsB = 0;
  mAltroCFG1 = 0;
  mAltroCFG2 = 0;
  mIsInitialized = false;
}

void RCUTrailer::constructFromRawPayload(const RawPayload& buffer)
{
  reset();
  auto payloadwords = buffer.getPayloadWords();
  int index = payloadwords.size() - 1;
  auto word = payloadwords[index];
  if ((word >> 30) != 3)
    throw Error(Error::ErrorType_t::DECODING_INVALID, "Last RCU trailer word not found!");
  mFirmwareVersion = (word >> 16) & 0xFF;

  mRCUId = (int)((word >> 7) & 0x1FF);
  int trailerSize = (word & 0x7F);

  if (trailerSize < 2)
    throw Error(Error::ErrorType_t::SIZE_INVALID, (boost::format("Invalid trailer size found (%d bytes) !") % (trailerSize * 4)).str().data());
  mTrailerSize = trailerSize;

  for (; trailerSize > 0; trailerSize--) {
    word = payloadwords[--index];
    if ((word >> 30) != 2) {
      std::cerr << "Missing RCU trailer identifier pattern!\n";
      continue;
    }
    int parCode = (word >> 26) & 0xF;
    int parData = word & 0x3FFFFFF;
    switch (parCode) {
      case 1:
        // ERR_REG1
        mFECERRA = ((parData >> 13) & 0x1FFF) << 7;
        mFECERRB = ((parData & 0x1FFF)) << 7;
        break;
      case 2:
        // ERR_REG2
        mERRREG2 = parData & 0x1FF;
        break;
      case 3:
        // ERR_REG3
        mERRREG3 = parData & 0x1FFFFFF;
        break;
      case 4:
        // FEC_RO_A
        mActiveFECsA = parData & 0xFFFF;
        break;
      case 5:
        // FEC_RO_B
        mActiveFECsB = parData & 0xFFFF;
        break;
      case 6:
        // RDO_CFG1
        mAltroCFG1 = parData & 0xFFFFF;
        break;
      case 7:
        // RDO_CFG2
        mAltroCFG2 = parData & 0x1FFFFFF;
        break;
      default:
        std::cerr << "Undefined parameter code " << parCode << ", ignore it !\n";
        break;
    }
  }
  mPayloadSize = payloadwords[--index] & 0x3FFFFFF;
  mIsInitialized = true;
}

double RCUTrailer::getTimeSample() const
{
  unsigned char fq = (mAltroCFG2 >> 5) & 0xF;
  double tSample;
  switch (fq) {
    case 0:
      // 20 MHz
      tSample = 2.0;
      break;
    case 1:
      // 10 Mhz
      tSample = 4.0;
      break;
    case 2:
      // 5 MHz
      tSample = 8.;
      break;
    default:
      throw Error(Error::ErrorType_t::SAMPLINGFREQ_INVALID, (boost::format("Invalid sampling frequency value %d !") % int(fq)).str().data());
  }

  return tSample * o2::constants::lhc::LHCBunchSpacingNS * 1.e-9;
}

double RCUTrailer::getL1Phase() const
{
  double tSample = getTimeSample(),
         phase = ((double)(mAltroCFG2 & 0x1F)) * o2::constants::lhc::LHCBunchSpacingNS * 1.e-9;
  if (phase >= tSample) {
    throw Error(Error::ErrorType_t::L1PHASE_INVALID, (boost::format("Invalid L1 trigger phase (%f >= %f) !") % phase % tSample).str().data());
  }
  return phase;
}

void RCUTrailer::printStream(std::ostream& stream) const
{
  stream << "RCU trailer (Format version 2):\n"
         << "==================================================\n"
         << "RCU ID:                                    " << mRCUId << "\n"
         << "Firmware version:                          " << int(mFirmwareVersion) << "\n"
         << "Trailer size:                              " << mTrailerSize << "\n"
         << "Payload size:                              " << mPayloadSize << "\n"
         << "FECERRA:                                   0x" << std::hex << mFECERRA << "\n"
         << "FECERRB:                                   0x" << std::hex << mFECERRB << "\n"
         << "ERRREG2:                                   0x" << std::hex << mERRREG2 << "\n"
         << "#channels skipped due to address mismatch: " << std::dec << getNumberOfChannelAddressMismatch() << "\n"
         << "#channels skipped due to bad block length: " << std::dec << getNumberOfChannelLengthMismatch() << "\n"
         << "Active FECs (branch A):                    0x" << std::hex << mActiveFECsA << "\n"
         << "Active FECs (branch B):                    0x" << std::hex << mActiveFECsB << "\n"
         << "Baseline corr:                             0x" << std::hex << int(getBaselineCorrection()) << "\n"
         << "Number of presamples:                      " << std::dec << int(getNumberOfPresamples()) << "\n"
         << "Number of postsamples:                     " << std::dec << int(getNumberOfPostsamples()) << "\n"
         << "Second baseline corr:                      " << (hasSecondBaselineCorr() ? "yes" : "no") << "\n"
         << "GlitchFilter:                              " << std::dec << int(getGlitchFilter()) << "\n"
         << "Number of non-ZS postsamples:              " << std::dec << int(getNumberOfNonZeroSuppressedPostsamples()) << "\n"
         << "Number of non-ZS presamples:               " << std::dec << int(getNumberOfNonZeroSuppressedPresamples()) << "\n"
         << "Number of ALTRO buffers:                   " << std::dec << getNumberOfAltroBuffers() << "\n"
         << "Number of pretrigger samples:              " << std::dec << int(getNumberOfPretriggerSamples()) << "\n"
         << "Number of samples per channel:             " << std::dec << getNumberOfSamplesPerChannel() << "\n"
         << "Sparse readout:                            " << (isSparseReadout() ? "yes" : "no") << "\n"
         << "Sampling time:                             " << std::scientific << getTimeSample() << " s\n"
         << "L1 Phase:                                  " << std::scientific << getL1Phase() << " s\n"
         << "AltroCFG1:                                 0x" << std::hex << mAltroCFG1 << "\n"
         << "AltroCFG2:                                 0x" << std::hex << mAltroCFG2 << "\n"
         << "==================================================\n"
         << std::dec << std::fixed;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::RCUTrailer& trailer)
{
  trailer.printStream(stream);
  return stream;
}