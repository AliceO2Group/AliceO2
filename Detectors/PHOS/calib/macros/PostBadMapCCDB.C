void PostBadMapCCDB()
{

  //Post test bad map for PHOS to test CCDB

  o2::ccdb::CcdbApi ccdb;
  map<string, string> metadata;               // do we want to store any meta data?
  ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation

  auto o2phosBM = new o2::phos::BadChannelMap();

  o2::phos::Geometry* geom = o2::phos::Geometry::GetInstance("Run3"); // Needed for tranforming 2D histograms to channel ID

  int nBad = 50;

  for (int i = 0; i < nBad; i++) {
    unsigned short channelID = gRandom->Uniform(56 * 64 * 3.5); //Random bad channels in 3.5 PHOS modules
    o2phosBM->addBadChannel(channelID);
  }

  ccdb.storeAsTFileAny(o2phosBM, "PHOS/BadMap", metadata, 1, 1670700184549); // one year validity time
}