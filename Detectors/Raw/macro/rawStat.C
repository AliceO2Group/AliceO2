#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Framework/Logger.h"
#include "DetectorsRaw/RawFileReader.h"
#include "CommonUtils/TreeStream.h"
#include "CommonUtils/TreeStreamRedirector.h"
#endif

/// Macro to write into the tree size per HBF contained in the raw data provided in the
/// RawFileReader config file.
/// No check for synchronization between different links is done

using namespace o2::raw;
using namespace o2::utils;

void rawStat(const std::string& conf)
{
  RawFileReader reader(conf);
  reader.init();
  int nLinks = reader.getNLinks();
  std::vector<char> buff;
  size_t hbfSize, cnt = 0;
  TreeStreamRedirector strm("hbfstat.root", "recreate");

  do {
    hbfSize = 0;
    int nonEmpty = 0;
    for (int il = 0; il < nLinks; il++) {
      auto& link = reader.getLink(il);
      auto sz = link.getNextHBFSize();
      if (sz) {
        nonEmpty++;
        hbfSize += sz;
        buff.resize(sz);
        link.readNextHBF(buff.data()); // just to advance
        strm << "links"
             << "id=" << il << "sz=" << sz << "\n";
      }
    }
    if (hbfSize) {
      strm << "hbfs"
           << "sz=" << hbfSize << "\n";
      LOG(INFO) << "hb " << cnt++ << " size: " << hbfSize << " in " << nonEmpty << " non-empty links";
    }
  } while (hbfSize);

  strm.Close();
}
