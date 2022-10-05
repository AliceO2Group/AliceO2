  
  //  in h : 
  // TVector3 posCkov(fTrkPos.X(), fTrkPos.Y(), zRad);



Methods using SetPhi / SetMag / SetTheta must be Polar3D or CylindricalEta3D
// pc, rad in findPhotCkov are XYZ

// fTrkDir, dirCkovTRS, dirCkovLORS XYZ

// dirTRS, dirLORS, dirCkov + refract -> dir need Polar3D


dir should be ok now; but input to refract? (her er dir brukt)

  // theta:
    // findPhotCkov -> fTrkDir
    // lors2Trs -> dirCkovTRS
    // trs2Lors -> dirCkovLORS
    // findRingCkov -> fTrkDir
  
  // setTheta
    // refract -> dir
  
  // Phi
    // findPhotCkov -> pc, rad : Ikke inputs 
    // lors2Trs -> fTrkDir, dirCkovTRS : Ikke inputs
    // trs2Lors -> fTrkDir, dirCkovLORS : Ikke inputs
    // intWithEdge -> fTrkDir : Ikke input
    
  //SetMagThetaPhi -> SetR; SetTheta; SetPhi -> Only Polar3D
    // dirTRS, dirLORS, dirCkov
  
  // in h : 
  //  dir.SetXYZ(-999, -999, -999);
  //  dir.SetTheta(TMath::ASin(sinref));
  // Theta, Phi, SetMagThetaPhi
