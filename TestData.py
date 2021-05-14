import ROOT
import array
import numpy as np

def get_variables():
# the full list of variables available for classification 
#    return ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
#            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
#            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
#            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
#            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
#            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
#            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]

# return list of variables to be used (eventually only a sub-set of the above)
    return ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]

def train(feature_filename = 'atlas-higgs-challenge-2014-v2_part.root', 
          TMVAoutfile = 'TMVAout.root'):
# this defines the algorithms and trainig options     

    # Open ROOT files and create TMVA factory
    feature_file = ROOT.TFile(feature_filename, 'read')
    output_file = ROOT.TFile(TMVAoutfile, 'recreate')
    for variable in get_variables():
        if variable in ['DER_mass_MMC', 'DER_mass_jet_jet', 'DER_deltaeta_jet_jet', 'DER_prodeta_jet_jet',\
         'DER_lep_eta_centrality', 'PRI_jet_leading_pt', 'PRI_jet_subleading_phi', 'PRI_jet_subleading_eta', \
         'PRI_jet_subleading_pt', 'PRI_jet_leading_phi', 'PRI_jet_leading_eta']:
            tmva_dataloader.AddVariable(variable, "F",-10,10)
        else:
            tmva_dataloader.AddVariable(variable, "F")
    print(feature_file)

if __name__ == '__main__':

	file = ROOT.TFile('atlas-higgs-challenge-2014-v2_part.root', 'read')
	#file = ROOT.TFile('TMVAout.root')
	feature_tree = file.Get("validation")
	feature_values = []

	for variable in get_variables():
		feature_values.append(array.array( 'f',  [0.]))
		feature_tree.SetBranchAddress(variable,  feature_values[-1])#feature_values[-1])
	#feature_tree.Print()
	#feature_tree.Scan('DER_mass_MMC')
	DER_mass_MMC = feature_tree.GetBranch("DER_mass_MMC")

	nevt = feature_tree.GetEntries()
	masses = []
	for evt in xrange(nevt):
		feature_tree.GetEntry(evt)
		if feature_values[0][0] != -999:
			masses.append(feature_values[0][0])

	print(np.mean(masses))
	print(np.max(masses))
