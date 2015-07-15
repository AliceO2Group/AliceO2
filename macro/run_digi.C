void run_digi(Int_t nEvents = 10, TString mcEngine = "TGeant3"){
        // Initialize logger
        FairLogger *logger = FairLogger::GetLogger();
        logger->SetLogVerbosityLevel("HIGH");
        logger->SetLogScreenLevel("DEBUG"); 
        
        // Input and output file name
        std::stringstream inputfile, outputfile, paramfile;
        inputfile << "AliceO2_" << mcEngine << ".mc_" << nEvents << "_event.root";
        paramfile << "AliceO2_" << mcEngine << ".params_" << nEvents << ".root";
        outputfile << "AliceO2_" << mcEngine << ".digi_" << nEvents << "_event.root";

        // Setup timer
        TStopwatch timer;

        // Setup FairRoot analysis manager
        FairRunAna * fRun = new FairRunAna;
        fRun->SetInputFile(inputfile.str().c_str());
        fRun->SetOutputFile(outputfile.str().c_str());

        // Setup Runtime DB
        FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
        FairParRootFileIo* parInput1 = new FairParRootFileIo();
        parInput1->open(paramfile.str().c_str());
        rtdb->setFirstInput(parInput1);

        TGeoManager::Import("geofile_full.root");

        // Setup digitizer
        AliceO2::ITS::Digitizer *digi = new AliceO2::ITS::Digitizer;
        fRun->AddTask(digi);

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

        std::cout << endl << std::endl;
        std::cout << "Output file is "    << outputfile.str() << std::endl;
        //std::cout << "Parameter file is " << parFile << std::endl;
        std::cout << "Real time " << rtime << " s, CPU time " << ctime
                  << "s" << endl << endl;
        std::cout << "Macro finished successfully." << std::endl;
}
