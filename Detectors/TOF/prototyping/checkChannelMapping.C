#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TOFBase/Geo.h"
#endif

using namespace o2::tof;

void checkChannelMapping()
{
  int nchannels = Geo::NCHANNELS;
  int volume[5];
  printf("N TOF channels = %d\n", nchannels);

  int failed1 = 0;
  int failed2 = 0;
  for (int i = 0; i < nchannels; i++) {
    int echan = Geo::getECHFromCH(i);
    if (i != Geo::getCHFromECH(echan)) {
      failed1++;
      printf("check 1)%d %d\n", i, Geo::getCHFromECH(echan));
    }

    volume[0] = Geo::getCrateFromECH(echan);
    volume[1] = Geo::getTRMFromECH(echan);
    volume[2] = Geo::getChainFromECH(echan);
    volume[3] = Geo::getTDCFromECH(echan);
    volume[4] = Geo::getTDCChFromECH(echan);

    if (echan != Geo::getECHFromIndexes(volume[0], volume[1], volume[2], volume[3], volume[4])) {
      failed2++;
      printf("check2) %d %d (%d %d %d %d %d)\n", echan, Geo::getECHFromIndexes(volume[0], volume[1], volume[2], volume[3], volume[4]), volume[0], volume[1], volume[2], volume[3], volume[4]);
    }
  }

  printf("Check1 failed = %d\n", failed1);
  printf("Check2 failed = %d\n", failed2);
}
