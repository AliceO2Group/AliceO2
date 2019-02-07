#include "AliGPUCADisplay.h"

const char* HelpText[] = {
	"[n] / [SPACE]                 Next event",
	"[q] / [Q] / [ESC]             Quit",
	"[r]                           Reset Display Settings",
	"[l] / [k] / [J]               Draw single slice (next  / previous slice), draw related slices (same plane in phi)",
	"[;] / [:]                     Show splitting of TPC in slices by extruding volume, [:] resets",
	"[#]                           Invert colors",
	"[y] / [Y] / ['] / [X] / [M]   Start Animation, Add / remove animation point, Reset, Cycle mode",
	"[>] / [<]                     Toggle config interpolation during animation / change animation interval (via movement)",
	"[g]                           Draw Grid",
	"[i]                           Project onto XY-plane",
	"[x]                           Exclude Clusters used in the tracking steps enabled for visualization ([1]-[8])",
	"[.]                           Exclude rejected tracks",
	"[c]                           Mark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20",
	"[B]                           Mark clusters attached as adjacent",
	"[L] / [K]                     Draw single collisions (next / previous)",
	"[C]                           Colorcode clusters of different collisions",
	"[v]                           Hide rejected clusters from tracks",
	"[b]                           Hide all clusters not belonging or related to matched tracks",
	"[j]                           Show global tracks as additional segments of final tracks",
	"[E] / [G]                     Extrapolate tracks / loopers",
	"[t] / [T]                     Take Screenshot / Record animation to pictures",
	"[Z]                           Change screenshot resolution (scaling factor)",
	"[S] / [A] / [D]               Enable or disable smoothing of points / smoothing of lines / depth buffer",
	"[W] / [U] / [V]               Toggle anti-aliasing (MSAA at raster level / change downsampling FSAA facot / toggle VSync",
	"[F] / [_] / [R]               Switch fullscreen / Maximized window / FPS rate limiter",
	"[I]                           Enable / disable GL indirect draw",
	"[o] / [p] / [O] / [P]         Save / restore current camera position / animation path",
	"[h]                           Print Help",
	"[H]                           Show info texts",
	"[w] / [s] / [a] / [d]         Zoom / Strafe Left and Right",
	"[pgup] / [pgdn]               Strafe Up and Down",
	"[e] / [f]                     Rotate",
	"[+] / [-]                     Make points thicker / fainter (Hold SHIFT for lines)",
	"[MOUSE 1]                     Look around",
	"[MOUSE 2]                     Shift camera",
	"[MOUSE 1+2]                   Zoom / Rotate",
	"[SHIFT]                       Slow Zoom / Move / Rotate",
	"[ALT] / [CTRL] / [m]          Focus camera on origin / orient y-axis upwards (combine with [SHIFT] to lock) / Cycle through modes",
	"[1] ... [8] / [N]             Enable display of clusters, preseeds, seeds, starthits, tracklets, tracks, global tracks, merged tracks / Show assigned clusters in colors"
	"[F1] / [F2]                   Enable / disable drawing of TPC / TRD"
	//FREE: u z
};

void AliGPUCADisplay::PrintHelp()
{
	infoHelpTimer.ResetStart();
	for (unsigned int i = 0;i < sizeof(HelpText) / sizeof(HelpText[0]);i++) printf("%s\n", HelpText[i]);
}

void AliGPUCADisplay::HandleKeyRelease(unsigned char key)
{
	if (key == mBackend->KEY_ENTER || key == 'n')
	{
		mBackend->displayControl = 1;
		SetInfo("Showing next event", 1);
	}
	else if (key == 27 || key == 'q' || key == 'Q' || key == mBackend->KEY_ESCAPE)
	{
		mBackend->displayControl = 2;
		SetInfo("Exiting", 1);
	}
	else if (key == 'r')
	{
		resetScene = 1;
		SetInfo("View reset", 1);
	}
	else if (key == mBackend->KEY_ALT && mBackend->keysShift[mBackend->KEY_ALT])
	{
		camLookOrigin ^= 1;
		cameraMode = camLookOrigin + 2 * camYUp;
		SetInfo("Camera locked on origin: %s", camLookOrigin ? "enabled" : "disabled");
	}
	else if (key == mBackend->KEY_CTRL && mBackend->keysShift[mBackend->KEY_CTRL])
	{
		camYUp ^= 1;
		cameraMode = camLookOrigin + 2 * camYUp;
		SetInfo("Camera locked on y-axis facing upwards: %s", camYUp ? "enabled" : "disabled");
	}
	else if (key == 'm')
	{
		cameraMode++;
		if (cameraMode == 4) cameraMode = 0;
		camLookOrigin = cameraMode & 1;
		camYUp = cameraMode & 2;
		const char* modeText[] = {"Descent (free movement)", "Focus locked on origin (y-axis forced upwards)", "Spectator (y-axis forced upwards)", "Focus locked on origin (with free rotation)"};
		SetInfo("Camera mode %d: %s", cameraMode, modeText[cameraMode]);
	}
	else if (key == mBackend->KEY_ALT)
	{
		mBackend->keys[mBackend->KEY_CTRL] = false; //Release CTRL with alt, to avoid orienting along y automatically!
	}
	else if (key == 'l')
	{
		if (cfg.drawSlice >= (cfg.drawRelatedSlices ? (fgkNSlices / 4 - 1) : (fgkNSlices - 1)))
		{
			cfg.drawSlice = -1;
			SetInfo("Showing all slices", 1);
		}
		else
		{
			cfg.drawSlice++;
			SetInfo("Showing slice %d", cfg.drawSlice);
		}
	}
	else if (key == 'k')
	{
		if (cfg.drawSlice <= -1)
		{
			cfg.drawSlice = cfg.drawRelatedSlices ? (fgkNSlices / 4 - 1) : (fgkNSlices - 1);
		}
		else
		{
			cfg.drawSlice--;
		}
		if (cfg.drawSlice == -1) SetInfo("Showing all slices", 1);
		else SetInfo("Showing slice %d", cfg.drawSlice);
	}
	else if (key == 'J')
	{
		cfg.drawRelatedSlices ^= 1;
		SetInfo("Drawing of related slices %s", cfg.drawRelatedSlices ? "enabled" : "disabled");
	}
	else if (key == 'L')
	{
		if (cfg.showCollision >= nCollisions - 1)
		{
			cfg.showCollision = -1;
			SetInfo("Showing all collisions", 1);
		}
		else
		{
			cfg.showCollision++;
			SetInfo("Showing collision %d", cfg.showCollision);
		}
	}
	else if (key == 'K')
	{
		if (cfg.showCollision <= -1)
		{
			cfg.showCollision = nCollisions - 1;
		}
		else
		{
			cfg.showCollision--;
		}
		if (cfg.showCollision == -1) SetInfo("Showing all collisions", 1);
		else SetInfo("Showing collision %d", cfg.showCollision);
	}
	else if (key == 'F')
	{
		fullscreen ^= 1;
		mBackend->SwitchFullscreen(fullscreen);
		SetInfo("Toggling full screen (%d)", (int) fullscreen);
	}
	else if (key == '_')
	{
		maximized ^= 1;
		mBackend->ToggleMaximized(maximized);
		SetInfo("Toggling maximized window (%d)", (int) maximized);
	}
	else if (key == 'R')
	{
		mBackend->maxFPSRate ^= 1;
		SetInfo("FPS rate %s", mBackend->maxFPSRate ? "not limited" : "limited");
	}
	else if (key == 'H')
	{
		printInfoText += 1;
		printInfoText &= 3;
		SetInfo("Info text display - console: %s, onscreen %s", (printInfoText & 2) ? "enabled" : "disabled", (printInfoText & 1) ? "enabled" : "disabled");
	}
	else if (key == 'j')
	{
		separateGlobalTracks ^= 1;
		SetInfo("Seperated display of global tracks %s", separateGlobalTracks ? "enabled" : "disabled");
		updateDLList = true;
	}
	else if (key == 'c')
	{
		if (markClusters == 0) markClusters = 1;
		else if (markClusters >= 0x20) markClusters = 0;
		else markClusters <<= 1;
		SetInfo("Cluster flag highlight mask set to %d (%s)", markClusters, markClusters == 0 ? "off" : markClusters == 1 ? "split pad" : markClusters == 2 ? "split time" : markClusters == 4 ? "edge" : markClusters == 8 ? "singlePad" : markClusters == 0x10 ? "reject distance" : "reject error");
		updateDLList = true;
	}
	else if (key == 'B')
	{
		markAdjacentClusters++;
		if (markAdjacentClusters == 5) markAdjacentClusters = 7;
		if (markAdjacentClusters == 9) markAdjacentClusters = 15;
		if (markAdjacentClusters == 18) markAdjacentClusters = 0;
		if (markAdjacentClusters == 17) SetInfo("Marking protected clusters (%d)", markAdjacentClusters);
		else if (markAdjacentClusters == 16) SetInfo("Marking removable clusters (%d)", markAdjacentClusters);
		else SetInfo("Marking adjacent clusters (%d): rejected %s, tube %s, looper leg %s, low Pt %s", markAdjacentClusters, markAdjacentClusters & 1 ? "yes" : " no", markAdjacentClusters & 2 ? "yes" : " no", markAdjacentClusters & 4 ? "yes" : " no", markAdjacentClusters & 8 ? "yes" : " no");
		updateDLList = true;
	}
	else if (key == 'C')
	{
		cfg.colorCollisions ^= 1;
		SetInfo("Color coding of collisions %s", cfg.colorCollisions ? "enabled" : "disabled");
	}
	else if (key == 'N')
	{
		cfg.colorClusters ^= 1;
		SetInfo("Color coding for seed / trrack attachmend %s", cfg.colorClusters ? "enabled" : "disabled");
	}
	else if (key == 'E')
	{
		cfg.propagateTracks += 1;
		if (cfg.propagateTracks == 4) cfg.propagateTracks = 0;
		const char* infoText[] = {"Hits connected", "Hits connected and propagated to vertex", "Reconstructed track propagated inwards and outwards", "Monte Carlo track"};
		SetInfo("Display of propagated tracks: %s", infoText[cfg.propagateTracks]);
	}
	else if (key == 'G')
	{
		propagateLoopers ^= 1;
		SetInfo("Propagation of loopers %s", propagateLoopers ? "enabled" : "disabled");
		updateDLList = true;
	}
	else if (key == 'v')
	{
		hideRejectedClusters ^= 1;
		SetInfo("Rejected clusters are %s", hideRejectedClusters ? "hidden" : "shown");
		updateDLList = true;
	}
	else if (key == 'b')
	{
		hideUnmatchedClusters ^= 1;
		SetInfo("Unmatched clusters are %s", hideRejectedClusters ? "hidden" : "shown");
		updateDLList = true;
	}
	else if (key == 'i')
	{
		projectxy ^= 1;
		SetInfo("Projection onto xy plane %s", projectxy ? "enabled" : "disabled");
		updateDLList = true;
	}
	else if (key == 'S')
	{
		cfg.smoothPoints ^= true;
		SetInfo("Smoothing of points %s", cfg.smoothPoints ? "enabled" : "disabled");
	}
	else if (key == 'A')
	{
		cfg.smoothLines ^= true;
		SetInfo("Smoothing of lines %s", cfg.smoothLines ? "enabled" : "disabled");
	}
	else if (key == 'D')
	{
		cfg.depthBuffer ^= true;
		GLint depthBits;
		glGetIntegerv(GL_DEPTH_BITS, &depthBits);
		SetInfo("Depth buffer (z-buffer, %d bits) %s", depthBits, cfg.depthBuffer ? "enabled" : "disabled");
		setDepthBuffer();
	}
	else if (key == 'W')
	{
		drawQualityMSAA*= 2;
		if (drawQualityMSAA < 2) drawQualityMSAA = 2;
		if (drawQualityMSAA > 16) drawQualityMSAA = 0;
		UpdateOffscreenBuffers();
		SetInfo("Multisampling anti-aliasing factor set to %d", drawQualityMSAA);
	}
	else if (key == 'U')
	{
		drawQualityDownsampleFSAA++;
		if (drawQualityDownsampleFSAA == 1) drawQualityDownsampleFSAA = 2;
		if (drawQualityDownsampleFSAA == 5) drawQualityDownsampleFSAA = 0;
		UpdateOffscreenBuffers();
		SetInfo("Downsampling anti-aliasing factor set to %d", drawQualityDownsampleFSAA);
	}
	else if (key == 'V')
	{
		drawQualityVSync ^= true;
		mBackend->SetVSync(drawQualityVSync);
		SetInfo("VSync: %s", drawQualityVSync ? "enabled" : "disabled");
	}
	else if (key == 'I')
	{
		useGLIndirectDraw ^= true;
		SetInfo("OpenGL Indirect Draw %s", useGLIndirectDraw ? "enabled" : "disabled");
		updateDLList = true;
	}
	else if (key == ';')
	{
		updateDLList = true;
		Xadd += 60;
		Zadd += 60;
		SetInfo("TPC sector separation: %f %f", Xadd, Zadd);
	}
	else if (key == ':')
	{
		updateDLList = true;
		Xadd -= 60;
		Zadd -= 60;
		if (Zadd < 0 || Xadd < 0) Zadd = Xadd = 0;
		SetInfo("TPC sector separation: %f %f", Xadd, Zadd);
	}
	else if (key == '#')
	{
		invertColors ^= 1;
	}
	else if (key == 'g')
	{
		cfg.drawGrid ^= 1;
		SetInfo("Fast Cluster Search Grid %s", cfg.drawGrid ? "shown" : "hidden");
	}
	else if (key == 'x')
	{
		cfg.excludeClusters ^= 1;
		SetInfo(cfg.excludeClusters ? "Clusters of selected category are excluded from display" : "Clusters are shown", 1);
	}
	else if (key == '.')
	{
		hideRejectedTracks ^= 1;
		SetInfo("Rejected tracks are %s", hideRejectedTracks ? "hidden" : "shown");
		updateDLList = true;
	}
	else if (key == '1')
	{
		cfg.drawClusters ^= 1;
	}
	else if (key == '2')
	{
		cfg.drawInitLinks ^= 1;
	}
	else if (key == '3')
	{
		cfg.drawLinks ^= 1;
	}
	else if (key == '4')
	{
		cfg.drawSeeds ^= 1;
	}
	else if (key == '5')
	{
		cfg.drawTracklets ^= 1;
	}
	else if (key == '6')
	{
		cfg.drawTracks ^= 1;
	}
	else if (key == '7')
	{
		cfg.drawGlobalTracks ^= 1;
	}
	else if (key == '8')
	{
		cfg.drawFinal ^= 1;
	}
	else if (key == mBackend->KEY_F1)
	{
		cfg.drawTPC ^= 1;
	}
	else if (key == mBackend->KEY_F2)
	{
		cfg.drawTRD ^= 1;
	}
	else if (key == 't')
	{
		printf("Taking screenshot\n");
		static int nScreenshot = 1;
		char fname[32];
		sprintf(fname, "screenshot%d.bmp", nScreenshot++);
		DoScreenshot(fname);
		SetInfo("Taking screenshot (%s)", fname);
	}
	else if (key == 'Z')
	{
		screenshot_scale += 1;
		if (screenshot_scale == 5) screenshot_scale = 1;
		SetInfo("Screenshot scaling factor set to %d", screenshot_scale);
	}
	else if (key == 'y' || key == 'T')
	{
		if ((animateScreenshot = key == 'T')) animationExport++;
		if (animateVectors[0].size() > 1)
		{
			startAnimation();
			SetInfo("Starting animation", 1);
		}
		else
		{
			SetInfo("Insufficient animation points to start animation", 1);
		}
	}
	else if (key == '>')
	{
		animationChangeConfig ^= 1;
		SetInfo("Interpolating visualization settings during animation %s", animationChangeConfig ? "enabled" : "disabled");
	}
	else if (key == 'Y')
	{
		setAnimationPoint();
		SetInfo("Added animation point (%d points, %6.2f seconds)", (int) animateVectors[0].size(), animateVectors[0].back());
	}
	else if (key == 'X')
	{
		resetAnimation();
		SetInfo("Reset animation points", 1);
	}
	else if (key == '\'')
	{
		removeAnimationPoint();
		SetInfo("Removed animation point", 1);
	}
	else if (key == 'M')
	{
		cfg.animationMode++;
		if (cfg.animationMode == 7) cfg.animationMode = 0;
		resetAnimation();
		if (cfg.animationMode == 6) SetInfo("Animation mode %d - Centered on origin", cfg.animationMode);
		else SetInfo("Animation mode %d - Position: %s, Direction: %s", cfg.animationMode, cfg.animationMode & 2 ? "Spherical (spherical rotation)" : cfg.animationMode & 4 ? "Spherical (Euler angles)" : "Cartesian", cfg.animationMode & 1 ? "Euler angles" : "Quaternion");
	}
	else if (key == 'o')
	{
		FILE *ftmp = fopen("glpos.tmp", "w+b");
		if (ftmp)
		{
			int retval = fwrite(&currentMatrix[0], sizeof(currentMatrix[0]), 16, ftmp);
			if (retval != 16) printf("Error writing position to file\n");
			else printf("Position stored to file\n");
			fclose(ftmp);
		}
		else
		{
			printf("Error opening file\n");
		}
		SetInfo("Camera position stored to file", 1);
	}
	else if (key == 'p')
	{
		GLfloat tmp[16];
		FILE *ftmp = fopen("glpos.tmp", "rb");
		if (ftmp)
		{
			int retval = fread(&tmp[0], sizeof(tmp[0]), 16, ftmp);
			if (retval == 16)
			{
				glMatrixMode(GL_MODELVIEW);
				glLoadMatrixf(tmp);
				glGetFloatv(GL_MODELVIEW_MATRIX, currentMatrix);
				printf("Position read from file\n");
			}
			else
			{
				printf("Error reading position from file\n");
			}
			fclose(ftmp);
		}
		else
		{
			printf("Error opening file\n");
		}
		SetInfo("Camera position loaded from file", 1);
	}
	else if (key == 'O')
	{
		FILE *ftmp = fopen("glanimation.tmp", "w+b");
		if (ftmp)
		{
			fwrite(&cfg, sizeof(cfg), 1, ftmp);
			int size = animateVectors[0].size();
			fwrite(&size, sizeof(size), 1, ftmp);
			for (int i = 0;i < 9;i++) fwrite(animateVectors[i].data(), sizeof(animateVectors[i][0]), size, ftmp);
			fwrite(animateConfig.data(), sizeof(animateConfig[0]), size, ftmp);
			fclose(ftmp);
		}
		else
		{
			printf("Error opening file\n");
		}
		SetInfo("Animation path stored to file %s", "glanimation.tmp");
	}
	else if (key == 'P')
	{
		FILE *ftmp = fopen("glanimation.tmp", "rb");
		if (ftmp)
		{
			int retval = fread(&cfg, sizeof(cfg), 1, ftmp);
			int size;
			retval += fread(&size, sizeof(size), 1, ftmp);
			for (int i = 0;i < 9;i++)
			{
				animateVectors[i].resize(size);
				retval += fread(animateVectors[i].data(), sizeof(animateVectors[i][0]), size, ftmp);
			}
			animateConfig.resize(size);
			retval += fread(animateConfig.data(), sizeof(animateConfig[0]), size, ftmp);
			fclose(ftmp);
			updateConfig();
		}
		else
		{
			printf("Error opening file\n");
		}
		SetInfo("Animation path loaded from file %s", "glanimation.tmp");
	}
	else if (key == 'h')
	{
		PrintHelp();
		SetInfo("Showing help text", 1);
	}
	/*else if (key == '#')
	{
		testSetting++;
		SetInfo("Debug test variable set to %d", testSetting);
		updateDLList = true;
	}*/
}

void AliGPUCADisplay::HandleSendKey(int key)
{
	//fprintf(stderr, "key %d '%c'\n", key, (char) key);

	bool shifted = key >= 'A' && key <= 'Z';
	int press = key;
	if (press >= 'a' && press <= 'z') press += 'A' - 'a';
	bool oldShift = mBackend->keysShift[press];
	mBackend->keysShift[press] = shifted;
	HandleKeyRelease(key);
	mBackend->keysShift[press] = oldShift;
	key = 0;
}

void AliGPUCADisplay::PrintGLHelpText(float colorValue)
{
	for (unsigned int i = 0;i < sizeof(HelpText) / sizeof(HelpText[0]);i++)
	{
		mBackend->OpenGLPrint(HelpText[i], 40.f, 35 + 20 * (1 + i), colorValue, colorValue, colorValue, infoHelpTimer.GetCurrentElapsedTime() >= 5 ? (6 - infoHelpTimer.GetCurrentElapsedTime()) : 1, false);
	}
}
