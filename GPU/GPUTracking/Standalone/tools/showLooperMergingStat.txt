new TCanvas("qpttglsign")
mergeloopers->Draw("tgl2:tgl1>>qpttglsign(100,-0.2,0.2,100,-0.2,0.2)", "labeleq>0.5 && qpt1 * tgl1 * qpt2 * tgl2 < 0 && fabsf(tgl1) < 0.2 && fabsf(tgl2) < 0.2", "colz")
new TCanvas("okxy")
mergeloopers->Draw("(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) >> okxy(100,0,40)", "labeleq > 0.5")
new TCanvas("okxy2")
mergeloopers->Draw("x1-x2:y1-y2 >> okxy(100,-10,10,100,-10,10)", "labeleq > 0.5")
new TCanvas("failxy")
mergeloopers->Draw("(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) >> failxy(100,0,40)", "labeleq < 0.5")
new TCanvas("oktgl")
mergeloopers->Draw("tgl1-tgl2 >> oktgl(100,0,40)", "labeleq > 0.5")
new TCanvas("failtgl")
mergeloopers->Draw("tgl1-tgl2 >> failtgl(100,0,40)", "labeleq < 0.5")
new TCanvas("okqpt")
mergeloopers->Draw("qpt1-qpt2 >> okqpt(100,0,40)", "labeleq > 0.5")
new TCanvas("failqpt")
mergeloopers->Draw("qpt1-qpt2 >> failqpt(100,0,40)", "labeleq < 0.5")

mergeloopers->Draw("2 * 3.1415 * 0.5*(fabs(tgl1) + fabs(tgl2)) * 1./(0.5*(fabsf(qpt1)+fabsf(qpt2))*0.001501)/(fabsf(refz2)-fabsf(refz1))>>test3(1000,0,5)", "labeleq > 0.5 && sameside==1", "colz")

mergeloopers->Draw("(fabsf(absz2)-fabsf(absz1))/(2 * 3.1415 * 0.5*(fabs(tgl1) + fabs(tgl2)) * 1./(0.5*(fabsf(qpt1)+fabsf(qpt2))*0.001501)):fmodf((asinf(snp1)+a1-asinf(snp2)-a2)/(2*3.1415)+5.5,1.)-0.5>>foo(300,-0.5,0.5,300,0,5)", "labeleq > 0.5 && sameside==1 && qpt1 * qpt2 > 0 &&  snp1 * snp2 > 0", "colz")

positive correction, 1 side, 2d:
mergeloopers2->Draw("(fabsf(refz2)-fabsf(refz1))/(2 * 3.1415 * 0.5*(fabs(tgl1) + fabs(tgl2)) * 1./(0.5*(fabsf(qpt1)+fabsf(qpt2))*0.001501)):(fmodf((asinf(snp1)+a1-asinf(snp2)-a2)/(2*3.1415)+5.5,1.)-0.5)>>foo(300,-0.5,0.5,300,0,5)", "labeleq > 0.5 && refz1*refz2>0 && refz1 * qpt1 * tgl1 > 0", "colz")
positive correction, 1 side, 1d:
mergeloopers2->Draw("(fabsf(refz2)-fabsf(refz1))/(2 * 3.1415 * 0.5*(fabs(tgl1) + fabs(tgl2)) * 1./(0.5*(fabsf(qpt1)+fabsf(qpt2))*0.001501))+(fmodf((asinf(snp1)+a1-asinf(snp2)-a2)/(2*3.1415)+5.5,1.)-0.5)>>hh(300,-1,5)", "labeleq > 0.5 && refz1*refz2>0 && refz1 * qpt1 * tgl1 > 0", "colz")
negative correction, 1 side, 2d:
mergeloopers2->Draw("(fabsf(refz2)-fabsf(refz1))/(2 * 3.1415 * 0.5*(fabs(tgl1) + fabs(tgl2)) * 1./(0.5*(fabsf(qpt1)+fabsf(qpt2))*0.001501)):(fmodf((asinf(snp1)+a1-asinf(snp2)-a2)/(2*3.1415)+5.5,1.)-0.5)>>foo(300,-0.5,0.5,300,0,5)", "labeleq > 0.5 && refz1*refz2>0 && refz1 * qpt1 * tgl1 < 0", "colz")
