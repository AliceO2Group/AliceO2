#include "build_geometry.C"

void o2sim()
{
  auto run = new FairRunSim();
  run->SetOutputFile("o2sim.root"); // Output file
  run->SetName("TGeant3");          // Transport engine

  // construct geometry / including magnetic field
  build_geometry(run);

  // setup generator
  auto primGen = new FairPrimaryGenerator();
  auto boxGen = new FairBoxGenerator(211, 10); /*protons*/
  boxGen->SetEtaRange(-0.9, 0.9);
  boxGen->SetPRange(0.1, 5);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kTRUE);
  primGen->AddGenerator(boxGen);
  run->SetGenerator(primGen);

  // run init
  run->Init();

  // runtime database
  bool kParameterMerged = true;
  auto rtdb = run->GetRuntimeDb();
  auto parOut = new FairParRootFileIo(kParameterMerged);
  parOut->open("o2sim_par.root");
  rtdb->setOutput(parOut);
  rtdb->saveOutput();
  rtdb->print();

  run->Run(1);
}
