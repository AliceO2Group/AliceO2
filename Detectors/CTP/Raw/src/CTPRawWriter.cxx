#include "CTPRaw/CTPRawWriter.h"
#include "DataFormatsCTP/CTPRawData.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::ctp;
ClassImp(CTPRawWriter);
/**
 * @brief CTPRawWriter::init
 * CTP has one CRU and 3 GBT links
 * - Interaction Record
 * - Trigger Class Record
 * - other: counters, HB map, ...
 * Code based on Raw/test/testRawReaderWriter.cxx
 */
void CTPRawWriter::init()
{
  mWriter.useRDHVersion(6);
  std::string outFileName = o2::utils::concat_string("ctp_testdata_", std::to_string(mcru), ".raw");
  mWriter.registerLink(mlink0, mcru, mlink0, 0, outFileName);
  mWriter.registerLink(mlink1, mcru, mlink1, 0, outFileName);
}

int o2::ctp::CTPRawWriter::createRawFromIRs()
{
  // Generate interactions
  std::vector<o2::InteractionTimeRecord> irs(1000);
  o2::steer::InteractionSampler irSampler;
  irSampler.setInteractionRate(15000); // ~1.5 interactions per orbit
  irSampler.init();
  irSampler.generateCollisionTimes(irs);
  //
  std::vector<char> buffer;
  //
  //uint32_t lastorbit=0;
  for (const auto& ir : irs) {
    InteractionTimeRecord intrec = ir;
    //ir.print();
    // Generate none zero suppressed data of CTP IRs
    uint64_t inputMask = (uint64_t)(gRandom->Uniform() * TriggerInputMask_d);
    //std::cout << "bcid:0x"<< std::hex<< intrec.bc << " Orbit:0x" << intrec.orbit << " inputMask:0x" <<inputMask << std::endl;
    CTPIRdigit digit{intrec.bc, inputMask};
    buffer.resize(o2::raw::RDHUtils::GBTWord);
    std::memcpy(buffer.data(), &digit, sizeof(CTPIRdigit));
    mWriter.addData(mlink0, mcru, mlink0, 0, ir, buffer);
    mWriter.addData(mlink1, mcru, mlink1, 0, ir, buffer);
  }
  mWriter.writeConfFile(mWriter.getOrigin().str, "RAWDATA", mConfigName);
}
