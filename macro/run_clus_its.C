#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <sstream>

  #include <TStopwatch.h>

  #include "FairLogger.h"
  #include "FairRunAna.h"
  #include "FairFileSource.h"
  #include "FairRuntimeDb.h"
  #include "FairParRootFileIo.h"
  #include "FairSystemInfo.h"

  #include "ITSReconstruction/TrivialClustererTask.h"
#endif

void run_clus_its(Int_t nEvents = 10, TString mcEngine = "TGeant3"){
        // Initialize logger
        FairLogger *logger = FairLogger::GetLogger();
        logger->SetLogVerbosityLevel("LOW");
        logger->SetLogScreenLevel("INFO");

        // Input and output file name
        std::stringstream inputfile, outputfile, paramfile;
        inputfile << "AliceO2_" << mcEngine << ".digi_" << nEvents << "_event.root";
        paramfile << "AliceO2_" << mcEngine << ".params_" << nEvents << ".root";
        outputfile << "AliceO2_" << mcEngine << ".clus_" << nEvents << "_event.root";

        // Setup timer
        TStopwatch timer;

        // Setup FairRoot analysis manager
        FairRunAna * fRun = new FairRunAna();
        FairFileSource *fFileSource = new FairFileSource(inputfile.str().c_str());
        fRun->SetSource(fFileSource);
        fRun->SetOutputFile(outputfile.str().c_str());

        // Setup Runtime DB
        FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
        FairParRootFileIo* parInput1 = new FairParRootFileIo();
        parInput1->open(paramfile.str().c_str());
        rtdb->setFirstInput(parInput1);

        // Setup clusterizer
        o2::ITS::TrivialClustererTask *clus = new o2::ITS::TrivialClustererTask;
        fRun->AddTask(clus);

        fRun->Init();

        timer.Start();
        fRun->Run();

        std::cout << std::endl << std::endl;

        // Extract the maximal used memory an add is as Dart measurement
        // This line is filtered by CTest and the value send to CDash
        FairSystemInfo sysInfo;
        Float_t maxMemory=sysInfo.GetMaxMemory();
        std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
        std::cout << maxMemory;
        std::cout << "</DartMeasurement>" << std::endl;

        timer.Stop();
        Double_t rtime = timer.RealTime();
        Double_t ctime = timer.CpuTime();

        Float_t cpuUsage=ctime/rtime;
        cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
        cout << cpuUsage;
        cout << "</DartMeasurement>" << endl;
        cout << endl << endl;
        std::cout << "Macro finished succesfully." << std::endl;
        std::cout << "Output file is "    << outputfile.str() << std::endl;
        //std::cout << "Parameter file is " << parFile << std::endl;
        std::cout << "Real time " << rtime << " s, CPU time " << ctime
                  << "s" << endl << endl;
}
