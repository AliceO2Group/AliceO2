#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TSystem.h>
#include <TMath.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TRandom.h>

#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairPrimaryGenerator.h"
#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"

#include "DetectorsPassive/Cave.h"
#include "Field/MagneticField.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTSimulation/Detector.h"

#endif

extern TSystem *gSystem;

void run_sim_mft(Int_t nEvents = 1, Int_t nMuons = 100, TString mcEngine = "TGeant3")
{

  printf("Run simulations: %d ev %d mu %s \n",nEvents,nMuons,mcEngine.Data());
  //return;

  gRandom->SetSeed(0);	

  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH",geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR",tut_configdir.Data());

  // Output file name
  char fileout[100];
  sprintf(fileout, "AliceO2_%s.mc_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString outFile = fileout;

  // Parameter file name
  char filepar[100];
  sprintf(filepar, "AliceO2_%s.params_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString parFile = filepar;

  // In general, the following parts need not be touched

  // Debug option
  gDebug = 0;

  // Timer
  TStopwatch timer;
  timer.Start();

  // Create simulation run
  FairRunSim* run = new FairRunSim();
  run->SetName(mcEngine);      // Transport engine
  run->SetOutputFile(outFile); // Output file
  FairRuntimeDb* rtdb = run->GetRuntimeDb();

  // Create media
  run->SetMaterials("media.geo"); // Materials

  // Create geometry
  o2::Passive::Cave* cave = new o2::Passive::Cave("CAVE");
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  o2::field::MagneticField field("field","field +5kG");
  run->SetField(&field);
  
  o2::MFT::Detector* mft = new o2::MFT::Detector();
  run->AddModule(mft);
  
  // Delete the segmentations from previous runs
  gSystem->Exec(" rm -f mftSegmentations.root ");

  // build ITS upgrade detector
  // pALPIDE3 15x30 mm^2  (X,Z) with 26.88 x 29.24 micron pitch
  const double kSensThick = 18e-4;
  const double kPitchZ = 29.24e-4;
  const double kPitchX = 26.88e-4;
  const int    kNRow   = 512; 
  const int    kNCol   = 1024;
  const double kSiThickIB = 50e-4;
  const double kSiThickOB = 50e-4;
  //  const double kSensThick = 120e-4;   // -> sensor Si thickness
  //
  const double kReadOutEdge = 0.12;   // width of the readout edge (passive bottom)
  const double kTopEdge = 37.44e-4;   // dead area on top
  const double kLeftRightEdge   = 29.12e-4; // width of passive area on left/right of the sensor
  
  // create segmentations
  o2::ITSMFT::SegmentationPixel* seg0 = new o2::ITSMFT::SegmentationPixel(
    0,           // segID (0:9)
    1,           // chips per module
    kNCol,       // ncols (total for module)
    kNRow,       // nrows
    kPitchX,     // default row pitch in cm
    kPitchZ,     // default col pitch in cm
    kSensThick,  // sensor thickness in cm
    -1,          // no special left col between chips
    -1,          // no special right col between chips
    kLeftRightEdge, // left
    kLeftRightEdge, // right
    kTopEdge, // top
    kReadOutEdge // bottom
    );           // see SegmentationPixel.h for extra options
  seg0->Store(o2::MFT::GeometryTGeo::getMFTSegmentationFileName());
  cout << "Print the pixel segmentation: " << endl;
  seg0->Print();
  //return;
  
  // Create PrimaryGenerator
  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
  FairBoxGenerator* boxGen = new FairBoxGenerator(13, nMuons);

  //boxGen->SetXYZ(0.,0.,0.);
  boxGen->SetThetaRange(170.0, 177.0);
  boxGen->SetPRange(4., 20.);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kTRUE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);
  
  run->Init();
  
  // Runtime database
  Bool_t kParameterMerged = kTRUE;
  FairParRootFileIo* parOut = new FairParRootFileIo(kParameterMerged);
  parOut->open(parFile.Data());
  rtdb->setOutput(parOut);
  rtdb->saveOutput();
  rtdb->print();
  
  run->Run(nEvents);
  run->CreateGeometryFile("geofile_mft.root");

  // Finish
  timer.Stop();

  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
  cout << "Output file is " << outFile << endl;
  cout << "Parameter file is " << parFile << endl;
  cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl << endl;

}
