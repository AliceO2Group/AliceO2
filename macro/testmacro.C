void testMacro() {
  // test objects from AliceO2
  AliceO2::BasicXYZEHit<float,float> hit;

  // test objects from FairRoot
  FairMCPoint *p = new FairMCPoint();

  // FairRunSim
  FairRunSim * run = new FairRunSim();

  FairParRootFileIo * io = new FairParRootFileIo(true);
}
