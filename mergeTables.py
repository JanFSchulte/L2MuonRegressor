import pandas as pd

muons = pd.read_csv('data/data_muons.txt')
segments = pd.read_csv("data/data_segments.txt")

muons.columns = ["Event_ID",'Muon_L2ID', 'Muon_L2_pt', 'Muon_L2_eta', 'Muon_L2_phi', 'Muon_L2_nSegments', 'Muon_gen_pt', 'Muon_gen_eta', 'Muon_gen_phi','Muon_L2_deltaPhiFirstLast','Muon_L2_deltaDirFirstLast','Muon_L2_deltaEtaFirstLast']
segments.columns = ["Event_ID",'ST_L2ID','ST_layerID1', 'ST_globalR1', 'ST_globalZ1', 'ST_phi1', 'ST_eta1', 'ST_deltaDir1', 'ST_deltaPhi1', 'ST_deltaEta1', 'ST_phiBend1', 'ST_etaBend1','ST_layerID2','ST_globalR2', 'ST_globalZ2', 'ST_phi2', 'ST_eta2', 'ST_deltaDir2', 'ST_deltaPhi2', 'ST_deltaEta2', 'ST_phiBend2', 'ST_etaBend2','ST_layerID3','ST_globalR3', 'ST_globalZ3', 'ST_phi3', 'ST_eta3', 'ST_deltaDir3', 'ST_deltaPhi3', 'ST_deltaEta3', 'ST_phiBend3', 'ST_etaBend3','ST_layerID4','ST_globalR4', 'ST_globalZ4', 'ST_phi4', 'ST_eta4', 'ST_deltaDir4', 'ST_deltaPhi4', 'ST_deltaEta4', 'ST_phiBend4', 'ST_etaBend4']

data = pd.merge(muons,segments,left_on = ["Event_ID","Muon_L2ID"],right_on = ["Event_ID","ST_L2ID"])

data.to_csv('data/L2SegmentsSmall_preprocessed.csv', index=False)
