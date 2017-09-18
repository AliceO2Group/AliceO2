// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/MetricsService.h"
#include "Framework/AlgorithmSpec.h"
#include "FairMQLogger.h"

#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairPrimaryGenerator.h"
#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"

#include "DetectorsPassive/Cave.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSSimulation/Detector.h"

using namespace o2::framework;

using DataHeader = o2::Header::DataHeader;

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;

double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

namespace o2 {
namespace workflows {

DataProcessorSpec sim_its_ALP3() {
  return {
    "sim_its_ALP3",
    Inputs{},
    Outputs{
      {"ITS", "HITS", OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      [](const ConfigParamRegistry &params, ServiceRegistry &services) {
        Int_t nEvents = 10;
        TString mcEngine = "TGeant3";

        TString dir = getenv("VMCWORKDIR");
        TString geom_dir = dir + "/Detectors/Geometry/";
        gSystem->Setenv("GEOMPATH",geom_dir.Data());


        TString tut_configdir = dir + "/Detectors/gconfig";
        gSystem->Setenv("CONFIG_DIR",tut_configdir.Data());

        // Output file name
        char fileout[100];
        sprintf(fileout, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
        TString outFile = fileout;

        // Parameter file name
        char filepar[100];
        sprintf(filepar, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
        TString parFile = filepar;

        FairRunSim *run = new FairRunSim();
        run->SetName(mcEngine);
        run->SetOutputFile(outFile); // Output file
        FairRuntimeDb* rtdb = run->GetRuntimeDb();

        // Create media
        run->SetMaterials("media.geo"); // Materials

        // Create geometry
        o2::Passive::Cave* cave = new o2::Passive::Cave("CAVE");
        cave->SetGeometryFileName("cave.geo");
        run->AddModule(cave);

        /*FairConstField field;
         field.SetField(0., 0., 5.); //in kG
         field.SetFieldRegion(-5000.,5000.,-5000.,5000.,-5000.,5000.); //in c
        */
        o2::field::MagneticField field("field","field +5kG");
        run->SetField(&field);

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
        //
        const int kNLr = 7;
        const int kNLrInner = 3;
        const int kBuildLevel = 0;
        enum { kRmn, kRmd, kRmx, kNModPerStave, kPhi0, kNStave, kNPar };
        // Radii are from last TDR (ALICE-TDR-017.pdf Tab. 1.1, rMid is mean value)
        const double tdr5dat[kNLr][kNPar] = {
          {2.24, 2.34, 2.67,  9., 16.42, 12}, // for each inner layer: rMin,rMid,rMax,NChip/Stave, phi0, nStaves
          {3.01, 3.15, 3.46,  9., 12.18, 16},
          {3.78, 3.93, 4.21,  9.,  9.55, 20},
          {-1,  19.6 ,   -1,  4.,  0.  , 24},  // for others: -, rMid, -, NMod/HStave, phi0, nStaves // 24 was 49
          {-1,  24.55, -1,    4.,  0.  , 30},  // 30 was 61
          {-1,  34.39, -1,    7.,  0.  , 42},  // 42 was 88
          {-1,  39.34, -1,    7.,  0.  , 48}   // 48 was 100
        };
        const int nChipsPerModule = 7; // For OB: how many chips in a row
        const double zChipGap = 0.01;  // For OB: gap in Z between chips
        const double zModuleGap = 0.01;// For OB: gap in Z between modules

        // Delete the segmentations from previous runs
        // FIXME: this should probably be done on each iteration?
        gSystem->Exec(" rm itsSegmentations.root ");

        // create segmentations:
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
        seg0->Store(o2::ITS::GeometryTGeo::getITSSegmentationFileName());
        seg0->Print();

        double dzLr, rLr, phi0, turbo;
        int nStaveLr, nModPerStaveLr;

        o2::ITS::Detector* its = new o2::ITS::Detector("ITS", kTRUE, kNLr);
        run->AddModule(its);

        its->setStaveModelIB(o2::ITS::Detector::kIBModel4);
        its->setStaveModelOB(o2::ITS::Detector::kOBModel2);

        const int kNWrapVol = 3;
        const double wrpRMin[kNWrapVol]  = { 2.1, 15.0, 32.0};
        const double wrpRMax[kNWrapVol]  = {14.0, 30.0, 46.0};
        const double wrpZSpan[kNWrapVol] = {70., 95., 200.};

        its->setNumberOfWrapperVolumes(kNWrapVol); // define wrapper volumes for layers

        for (int iw = 0; iw < kNWrapVol; iw++) {
          its->defineWrapperVolume(iw, wrpRMin[iw], wrpRMax[iw], wrpZSpan[iw]);
        }

        for (int idLr = 0; idLr < kNLr; idLr++) {
          rLr = tdr5dat[idLr][kRmd];
          phi0 = tdr5dat[idLr][kPhi0];

          nStaveLr = TMath::Nint(tdr5dat[idLr][kNStave]);
          nModPerStaveLr = TMath::Nint(tdr5dat[idLr][kNModPerStave]);
          int nChipsPerStaveLr = nModPerStaveLr;
          if (idLr >= kNLrInner) {
            double modlen = nChipsPerModule*seg0->Dz() + (nChipsPerModule-1)*zChipGap;
            double zlen = nModPerStaveLr*modlen + (nModPerStaveLr-1)*zModuleGap;
            its->defineLayer(idLr, phi0, rLr, zlen, nStaveLr, nModPerStaveLr,
                             kSiThickOB, seg0->Dy(), seg0->getChipTypeID(), kBuildLevel);
            //      printf("Add Lr%d: R=%6.2f DZ:%6.2f Staves:%3d NMod/Stave:%3d\n",
            //	     idLr,rLr,nChipsPerStaveLr*seg0->Dz(),nStaveLr,nModPerStaveLr);
          } else {
            turbo = -radii2Turbo(tdr5dat[idLr][kRmn], rLr, tdr5dat[idLr][kRmx], seg0->Dx());
            its->defineLayerTurbo(idLr, phi0, rLr, nChipsPerStaveLr * seg0->Dz(), nStaveLr,
                                  nChipsPerStaveLr, seg0->Dx(), turbo, kSiThickIB, seg0->Dy(),
                                  seg0->getChipTypeID(), kBuildLevel);
            //      printf("Add Lr%d: R=%6.2f DZ:%6.2f Turbo:%+6.2f Staves:%3d NMod/Stave:%3d\n",
            //	     idLr,rLr,nChipsPerStaveLr*seg0->Dz(),turbo,nStaveLr,nModPerStaveLr);
          }
        }

        // Create PrimaryGenerator
        FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
        FairBoxGenerator* boxGen = new FairBoxGenerator(211, 100); //pions

        //boxGen->SetThetaRange(0.0, 90.0);
        boxGen->SetEtaRange(-0.9,0.9);
        boxGen->SetPtRange(1, 1.01);
        boxGen->SetPhiRange(0., 360.);
        boxGen->SetDebug(kFALSE);

        primGen->AddGenerator(boxGen);

        run->SetGenerator(primGen);

        // Initialize simulation run
        run->Init();

        // Runtime database
        Bool_t kParameterMerged = kTRUE;
        FairParRootFileIo* parOut = new FairParRootFileIo(kParameterMerged);
        parOut->open(parFile.Data());
        rtdb->setOutput(parOut);
        rtdb->saveOutput();
        rtdb->print();

        // This is the actual inner loop for the device
        return [run](const std::vector<DataRef> inputs,
                  ServiceRegistry& s,
                  DataAllocator& allocator) {
                 run->Run(10);
                 // FIXME: After we run we should readback events
                 // and push them as messages, for the next stage of
                 // processing.
              };
        }
      }
    };
  };
} // namespace workflows
} // namespace o2
