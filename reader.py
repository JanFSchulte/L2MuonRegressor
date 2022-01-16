import ROOT
import argparse
from math import *
import os



variables_muons = ["Event_ID",'Muon_L2ID', 'Muon_L2_pt', 'Muon_L2_eta', 'Muon_L2_phi', 'Muon_L2_nSegments', 'Muon_gen_pt', 'Muon_gen_eta', 'Muon_gen_phi','Muon_L2_deltaPhiFirstLast','Muon_L2_deltaDirFirstLast']

variables_stations = ["Event_ID",'ST_L2ID','ST_layerID', 'ST_previousLayerID', 'ST_globalR', 'ST_globalZ', 'ST_phi', 'ST_deltaDir', 'ST_deltaPhi']


counter = 0

out_muons = open('data/data_muons.txt', 'w')
out_segments = open('data/data_segments.txt', 'w')

f = "data/segmentsForML.root"
tree = ROOT.TChain()
tree.Add(f+"/hltSegmentAnalyzerForML/t")

counter = 0		
for ev in tree:
	offset = 0
	for i in range(0,ev.nL2s):
		if  (ev.nSegments[i] ==4 and ev.gen_eta[i] < 0.8): 
			deltaPhiFirstLast = abs(ev.segment_phi[offset] - ev.segment_phi[offset + ev.nSegments[i]-1])
			deltaEtaFirstLast = abs(ev.segment_eta[offset] - ev.segment_eta[offset + ev.nSegments[i]-1])
			dDirFirstLast = (ev.segment_globalDX[offset]*ev.segment_globalDX[offset + ev.nSegments[i]-1] + ev.segment_globalDY[offset]*ev.segment_globalDY[offset + ev.nSegments[i]-1] + ev.segment_globalDZ[offset]*ev.segment_globalDZ[offset + ev.nSegments[i]-1] )

			line = str(counter) + ',\t' + str(i) + ',\t' + str(ev.l2_pt[i]) + ',\t' + str(ev.l2_eta[i]) + ',\t' + str(ev.l2_phi[i]) + ',\t ' + str(ev.nSegments[i]) + ',\t' + str(ev.gen_pt[i]) + ',\t ' + str(ev.gen_eta[i]) + ',\t' + str(ev.gen_phi[i]) + ',\t' + str(deltaPhiFirstLast) + ',\t' + str(deltaEtaFirstLast) + ',\t'+ str(dDirFirstLast) + '\t'
			out_muons.write(line + '\n')
			line = str(counter) + ',\t' + str(i) + ',\t'
			for k in range(offset, offset+ev.nSegments[i]):
				line += str(ev.segment_layerID[k]) + ',\t' + str(ev.segment_globalR[k]) + ',\t' + str(ev.segment_globalZ[k]) + ',\t' +  str(ev.segment_phi[k]) + ',\t' +  str(ev.segment_eta[k]) + ',\t' + str(ev.segment_deltaDir[k]) + ',\t' + str(abs(ev.segment_deltaPhi[k])) + ',\t' + str(abs(ev.segment_deltaEta[k])) + ',\t'+ str(abs(ev.segment_phiBend[k])) + ',\t' + str(abs(ev.segment_etaBend[k])) + ',\t'


			#if ev.nSegments[i] < 7:
			#	line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + '\t'
			#if ev.nSegments[i] < 6:
			#	line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + '\t'
			#if ev.nSegments[i] < 5:
			#	line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + '\t'
			#if ev.nSegments[i] < 4:
			#	line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t'
			#if ev.nSegments[i] < 3:
			#	line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t'
			#if ev.nSegments[i] < 2:
			#	line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t'
			#if ev.nSegments[i] < 1:
		#		line  += str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t' +  str(-9999) + ',\t' + str(-9999) + ',\t' + str(-9999) + ',\t'

			


			line = line.rstrip(',\t')

			out_segments.write(line + '\n')		
		offset += ev.nSegments[i] 
	counter += 1
	
	if counter > 100000: break

out_muons.close()
out_segments.close()
