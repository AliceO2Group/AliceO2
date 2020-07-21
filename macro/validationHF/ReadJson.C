char* GetJsonString(const char* jsonFileName, const char* key){
  FILE* fj=fopen(jsonFileName,"r");
  char line[500];
  char* value=0x0;
  while(!feof(fj)){
    fgets(line,500,fj);
    if(strstr(line,key)){
      value=strtok(line, ":");
      value=strtok(NULL, ":");
      break;
    }
  }
  fclose(fj);
  return value;
}

int GetJsonInteger(const char* jsonFileName, const char* key){
  FILE* fj=fopen(jsonFileName,"r");
  char line[500];
  int value=-999;
  while(!feof(fj)){
    fgets(line,500,fj);
    if(strstr(line,key)){
      char* token=strtok(line, ":");
      token=strtok(NULL, ":");
      TString temp=token;
      temp.ReplaceAll("\"","");
      temp.ReplaceAll(",","");
      value=temp.Atoi();
      break;
    }
  }
  fclose(fj);
  return value;
}

bool GetJsonBool(const char* jsonFileName, const char* key){
  FILE* fj=fopen(jsonFileName,"r");
  char line[500];
  bool value=false;
  while(!feof(fj)){
    fgets(line,500,fj);
    if(strstr(line,key)){
      char* token=strtok(line, ":");
      token=strtok(NULL, ":");
      TString temp=token;
      temp.ReplaceAll("\"","");
      temp.ReplaceAll(",","");
      if(temp.Contains("true")) value=true;
      break;
    }
  }
  fclose(fj);
  return value;
}

float GetJsonFloat(const char* jsonFileName, const char* key){
  FILE* fj=fopen(jsonFileName,"r");
  char line[500];
  float value=-999.;
  while(!feof(fj)){
    fgets(line,500,fj);
    if(strstr(line,key)){
      char* token=strtok(line, ":");
      token=strtok(NULL, ":");
      TString temp=token;
      temp.ReplaceAll("\"","");
      temp.ReplaceAll(",","");
      value=temp.Atof();
      break;
    }
  }
  fclose(fj);
  return value;
}


void ReadJson(){
  char* value=GetJsonString("dpl-config_std.json","aod-file");
  printf("%s\n", value);  
  int i3p=GetJsonInteger("dpl-config_std.json","do3prong");
  printf("%d\n", i3p);  
  int tin=GetJsonInteger("dpl-config_std.json","triggerindex");
  printf("%d\n", tin);  
  float minpt=GetJsonFloat("dpl-config_std.json","ptmintrack");
  printf("%f\n", minpt);
  float dcatoprimxymin=GetJsonFloat("dpl-config_std.json","dcatoprimxymin");
  printf("%f\n", dcatoprimxymin);
  bool doit=GetJsonBool("dpl-config_std.json","b_propdca");
  printf("%d\n", doit);
  float  chi2=GetJsonFloat("dpl-config_std.json","d_minrelchi2change");
  printf("%f\n", chi2);
}
