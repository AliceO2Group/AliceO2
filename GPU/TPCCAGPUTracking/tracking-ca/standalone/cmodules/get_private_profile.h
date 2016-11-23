#ifndef GET_PRIVATE_PROFILE_H
#define GET_PRIVATE_PROFILE_H

static inline longint GetPrivateProfileString(char* lpSectionName, char* lpKeyName, char* lpDefault, char* lpBuffer, DWORD size, char* configfile)
{
	for (size_t i = 0;i < strlen(configfile);i++) if (configfile[i] == '\\') configfile[i] = '/';
	FILE* cfgfile = fopen(configfile, "r");
	if (cfgfile == NULL)
	{
		fprintf(stderr, "Error opening file %s\n", configfile);
		return(-1);
	}
	char linebuffer[1024];
	bool correctsection = false;
	//fprintf(stderr, "Searching for %s in %s default %s\n", lpKeyName, lpSectionName, lpDefault);
	while (!feof(cfgfile))
	{
		if (fgets(linebuffer, 1023, cfgfile) == NULL) break;
		if (linebuffer[0] == '[')
		{
			correctsection = strncmp(&linebuffer[1], lpSectionName, strlen(lpSectionName)) == 0 && linebuffer[strlen(lpSectionName) + 1] == ']';
			continue;
		}
		if (!correctsection) continue;
		if (strncmp(linebuffer, lpKeyName, strlen(lpKeyName)) == 0)
		{
			char* tmpptr = &linebuffer[strlen(lpKeyName)];
			while (*tmpptr == ' ') tmpptr++;
			if (*tmpptr != '=') continue;
			while (*(++tmpptr) == ' ') ;
			char* tmpptr2 = tmpptr;
			while (*tmpptr2 != 0 && *tmpptr2 != 10 && *tmpptr2 != 13) tmpptr2++;
			*tmpptr2 = 0;
			strncpy(lpBuffer, &tmpptr[0], size < strlen(tmpptr) ? size : strlen(tmpptr));
			lpBuffer[size < strlen(tmpptr) ? size : strlen(tmpptr)] = 0;
			fclose(cfgfile);
			//fprintf(stderr, "Found: %s in %s: '%s'\n", lpKeyName, lpSectionName, lpBuffer);
			return(strlen(tmpptr));
		}
	}
	if (lpDefault == NULL) *lpBuffer = 0;
	else
	{
		strncpy(lpBuffer, lpDefault, size < strlen(lpDefault) ? size : strlen(lpDefault));
		lpBuffer[size < strlen(lpDefault) ? size : strlen(lpDefault)] = 0;
	}
	fclose(cfgfile);
	//fprintf(stderr, "Not found: %s in %s, using default: '%s' -> '%s'\n", lpKeyName, lpSectionName, lpDefault, lpBuffer);
	return(strlen(lpDefault));
}

static inline longint GetPrivateProfileInt(char* lpSectionName, char* lpKeyName, int nDefault, char* configfile)
{
	char linebuffer[16] = "0";
	char strdefault[16];
	sprintf(strdefault, "%d", nDefault);
	GetPrivateProfileString(lpSectionName, lpKeyName, strdefault, linebuffer, 15, configfile);
	return(atoi(linebuffer));
}

static inline int GetPrivateProfileSectionNames(char* buffer, int buffersize, char* filename)
{
	for (size_t i = 0;i < strlen(filename);i++) if (filename[i] == '\\') filename[i] = '/';
	FILE* cfgfile = fopen(filename, "r");
	if (cfgfile == NULL)
	{
		fprintf(stderr, "Error opening file %s\n", filename);
		return(-1);
	}
	char linebuffer[1024];
	int nwritten = 0;
	while (!feof(cfgfile))
	{
		if (fgets(linebuffer, 1023, cfgfile) == NULL) break;
		char* tmpptr = linebuffer;
		while (*tmpptr == ' ') tmpptr++;
		if (*tmpptr != '[') continue;
		char* sectptr = ++tmpptr;
		int section_len = 0;
		while (*tmpptr && *tmpptr != 10 && *tmpptr != 13)
		{
			if (*tmpptr != ']')
			{
				tmpptr++;
				section_len++;
			}
			else
			{
				if (nwritten + section_len + 2 < buffersize)
				{
					memcpy(&buffer[nwritten], sectptr, section_len);
					buffer[nwritten + section_len] = 0;
					buffer[nwritten + section_len + 1] = 0;
				}
				nwritten += section_len + 1;
				break;
			}
		}
	}
	return(nwritten);
}

#endif
