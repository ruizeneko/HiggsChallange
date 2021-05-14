import ROOT
import array
import numpy as np
import sys
import csv
import pandas as pd

def savethisifneededlater():
    DER_mass_MMC = []
    DER_mass_transverse_met_lep = []
    DER_mass_vis = []
    DER_pt_h = []
    DER_deltaeta_jet_jet = []
    DER_mass_jet_jet = []
    DER_prodeta_jet_jet = []
    DER_deltar_tau_lep = []
    DER_pt_tot = []
    DER_sum_pt = []
    DER_pt_ratio_lep_tau = []
    DER_met_phi_centrality = []
    DER_lep_eta_centrality = []
    PRI_tau_pt = []
    PRI_tau_eta = []
    PRI_tau_phi = []
    PRI_lep_pt = []
    PRI_lep_eta = []
    PRI_lep_phi = []
    PRI_met = []
    PRI_met_phi = []
    PRI_met_sumet = []
    PRI_jet_num = []
    PRI_jet_leading_pt = []
    PRI_jet_leading_eta = []
    PRI_jet_leading_phi = []
    PRI_jet_subleading_pt = []
    PRI_jet_subleading_eta = []
    PRI_jet_subleading_phi = []
    PRI_jet_all_pt = []
    return -1

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

def get_all_variables():
    return ["EventId", "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"
            ]#"KaggleSet", "KaggleWeight"]

def GetRelevantIndices(VariableNames = None):
    # Input: array of Strings of the variables, for which data should be omitted with values of -999
    # output: numpy array object with the indices get_variables() corresponding to the strings
    if VariableNames is None:
        print('Did not receive any input for GetRelevantIndices!')
        return -1
    returnlist = np.array([], dtype = int)
    for name in VariableNames:
        tmp = get_all_variables().index(name)
        returnlist = np.append(returnlist, int(tmp))
    return returnlist

def extractDataFromRoot(line = None):
    if line is None:
        print('Did not get any values. Returning')
        return -1
    returnvalues = []
    for i in line:
        returnvalues.append(i[0])
    return returnvalues

def get_del_variables():
    return ['PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi',
    'PRI_jet_leading_pt','DER_lep_eta_centrality','DER_prodeta_jet_jet','DER_mass_MMC','DER_deltaeta_jet_jet','DER_mass_jet_jet']

if __name__ == '__main__':
    if sys.argv[1] == 'RootToCsv':
        file = ROOT.TFile('atlas-higgs-challenge-2014-v2.root', 'read')

        file.ls()
        # need to iterate over signal, background, validation, e.g.: file.Get("signal")

        validation_tree = file.Get("validation")
        validation_values = []
        #validation_tree.Print()
        #validation_tree.Show(5)
        for variable in get_all_variables():
            validation_values.append(array.array( 'f',  [0.]))
            validation_tree.SetBranchAddress(variable,  validation_values[-1])#validation_values[-1])
        label_value = array.array( 'c',  ['x'])
        weight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        validation_tree.SetBranchAddress('Label',  label_value)
        validation_tree.SetBranchAddress('Weight',  weight_value)
        kaggleset_value = array.array( 'c',  ['x'])
        kaggleweight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        validation_tree.SetBranchAddress('KaggleSet',  kaggleset_value)
        validation_tree.SetBranchAddress('KaggleWeight',  kaggleweight_value)

        validation_values.append(label_value)
        validation_values.append(weight_value)
        validation_values.append(kaggleset_value)
        validation_values.append(kaggleweight_value)

        validation_nevt = validation_tree.GetEntries()
        print('Received %i events for Validation Tree!' % validation_nevt)
        data = []
        for evt in xrange(validation_nevt):
            validation_tree.GetEntry(evt)
            data.append(extractDataFromRoot(validation_values))
        print('Saving Validation to CSV')
        val_df = pd.DataFrame(data)
        val_name = '/home/ubedl/Uni/HiggsChallenge/atlas-higgs-challenge-2014_validation_full.csv'
        head = get_all_variables()
        print(head)
        for mn in ['Label', 'Weight', 'KaggleSet', 'KaggleWeight']:
            head.append(mn)
        #head = head.append(['Label','Weight','KaggleSet','KaggleWeight'])
        print(head)
        val_df.to_csv(val_name, sep=',', header = head)
        print('Validation Done')

        signal_tree = file.Get("signal")
        signal_values = []
        #signal_tree.Print()
        #signal_tree.Show(5)
        for variable in get_all_variables():
            signal_values.append(array.array( 'f',  [0.]))
            signal_tree.SetBranchAddress(variable,  signal_values[-1])#signal_values[-1])
        label_value = array.array( 'c',  ['x'])
        weight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        signal_tree.SetBranchAddress('Label',  label_value)
        signal_tree.SetBranchAddress('Weight',  weight_value)
        kaggleset_value = array.array( 'c',  ['x'])
        kaggleweight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        signal_tree.SetBranchAddress('KaggleSet',  kaggleset_value)
        signal_tree.SetBranchAddress('KaggleWeight',  kaggleweight_value)

        signal_values.append(label_value)
        signal_values.append(weight_value)
        signal_values.append(kaggleset_value)
        signal_values.append(kaggleweight_value)
        signal_nevt = signal_tree.GetEntries()
        print('Received %i events for Signal Tree!' % signal_nevt)
        data = []
        for evt in xrange(signal_nevt):
            signal_tree.GetEntry(evt)
            data.append(extractDataFromRoot(signal_values))
        print('Saving Signal to CSV')
        val_df = pd.DataFrame(data)
        val_name = '/home/ubedl/Uni/HiggsChallenge/atlas-higgs-challenge-2014_signal_full.csv'
        val_df.to_csv(val_name, sep=',', header = head)
        print('Signal Done')

        background_tree = file.Get("background")
        background_values = []
        #background_tree.Print()
        #background_tree.Show(5)
        for variable in get_all_variables():
            background_values.append(array.array( 'f',  [0.]))
            background_tree.SetBranchAddress(variable,  background_values[-1])#background_values[-1])
        label_value = array.array( 'c',  ['x'])
        weight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        background_tree.SetBranchAddress('Label',  label_value)
        background_tree.SetBranchAddress('Weight',  weight_value)
        kaggleset_value = array.array( 'c',  ['x'])
        kaggleweight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        background_tree.SetBranchAddress('KaggleSet',  kaggleset_value)
        background_tree.SetBranchAddress('KaggleWeight',  kaggleweight_value)

        background_values.append(label_value)
        background_values.append(weight_value)
        background_values.append(kaggleset_value)
        background_values.append(kaggleweight_value)
        background_nevt = background_tree.GetEntries()
        print('Received %i events for Background Tree!' % background_nevt)
        data = []
        for evt in xrange(background_nevt):
            background_tree.GetEntry(evt)
            data.append(extractDataFromRoot(background_values))
        print('Saving Background to CSV')
        val_df = pd.DataFrame(data)
        val_name = '/home/ubedl/Uni/HiggsChallenge/atlas-higgs-challenge-2014_background_full.csv'
        val_df.to_csv(val_name, sep=',', header = head)
        print('Background Done')




    elif sys.argv[1] == 'ReduceData':
        file = ROOT.TFile('atlas-higgs-challenge-2014-v2_part.root', 'read')
        output_file = ROOT.TFile('Dataset_ModSig.root', 'recreate')
        file.ls()
        for TreeName in ['signal', 'background', 'validation']:
            validation_tree_out = ROOT.TTree(TreeName, TreeName)
            
            #file = ROOT.TFile('TMVAout.root')
            # need to iterate over signal, background, validation, e.g.: file.Get("signal")
            validation_tree = file.Get(TreeName)
            validation_values = []

            for variable in get_all_variables():
                validation_values.append(array.array( 'f',  [0.]))
                validation_tree.SetBranchAddress(variable,  validation_values[-1])#validation_values[-1])

            label_value = array.array( 'c',  ['x'])
            weight_value = array.array( 'f',  [0.0])
        
            # Add label and weight
            validation_tree.SetBranchAddress('Label',  label_value)
            validation_tree.SetBranchAddress('Weight',  weight_value)
            kaggleset_value = array.array( 'c',  ['x'])
            kaggleweight_value = array.array( 'f',  [0.0])
        
            # Add label and weight
            validation_tree.SetBranchAddress('KaggleSet',  kaggleset_value)
            validation_tree.SetBranchAddress('KaggleWeight',  kaggleweight_value)

            validation_values.append(label_value)
            validation_values.append(weight_value)
            validation_values.append(kaggleset_value)
            validation_values.append(kaggleweight_value)

            #validation_tree.Print()
            #validation_tree.Scan('DER_mass_MMC')
            #validation_tree.Scan('KaggleSet')
            # indices is a list of indices for which we want to check the values.
            indices = GetRelevantIndices(get_del_variables())
            #print(indices)
            nevt = validation_tree.GetEntries()
            data = []

            # iterate over all events. if any entry in the specified indices list is -999 discard that event.
            # otherwise write it to new file. Thats the basic idea
            for evt in xrange(nevt):
                validation_tree.GetEntry(evt)
                if -999 in np.array([validation_values[index] for index in indices]) and TreeName in ["signal"]:
                    continue
                else:
                    data.append(extractDataFromRoot(validation_values))
            validation_out_values = []
            for variable in get_all_variables():
                validation_out_values.append(array.array( 'f',  [ 0. ]))
                validation_tree_out.Branch(variable, validation_out_values[-1], variable + "/F")

            validation_out_values.append(array.array('c', ['x']))
            validation_tree_out.Branch('Label', validation_out_values[-1], 'Label/C')
            validation_out_values.append(array.array('f', [0.0]))
            validation_tree_out.Branch('Weight', validation_out_values[-1], 'Weight/F')
            validation_out_values.append(array.array('c', ['x']))
            validation_tree_out.Branch('KaggleSet', validation_out_values[-1], 'KaggleSet/C')
            validation_out_values.append(array.array('f', [0.0]))
            validation_tree_out.Branch('KaggleWeight', validation_out_values[-1], 'KaggleWeight/F')

            print(len(data[0]))
            print(len(validation_out_values))
            #print(data[:])
            for idx, line in enumerate(data[:]):
                #print(validation_out_values)
                for i,val in enumerate(line):
                    #print(i,val)
                    validation_out_values[i][0] = val#array.array('f',[val])
                #print(validation_out_values)
                validation_tree_out.Fill()
            output_file.Write()


        output_file.Close()
        #print(np.array([validation_values[index] for index in [0,1,2,3,4,5]]).flatten())
        #print(np.mean(masses))
        #print(np.max(masses))
        #print(np.min(masses))

    elif sys.argv[1] == 'CheckReducedData':
        file = ROOT.TFile('TestData_MW_2.root', 'read')
        file.ls()
        #file = ROOT.TFile('TMVAout.root')
        # need to iterate over signal, background, validation, e.g.: file.Get("signal")
        validation_tree = file.Get("validation")
        validation_values = []
        validation_tree.Print()

        for variable in get_all_variables():
            validation_values.append(array.array( 'f',  [ 0. ]))
            validation_tree.SetBranchAddress(variable,  validation_values[-1])#validation_values[-1])
        label_value = array.array( 'c',  ['x'])
        weight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        validation_tree.SetBranchAddress('Label',  label_value)
        validation_tree.SetBranchAddress('Weight',  weight_value)
        kaggleset_value = array.array( 'c',  ['x'])
        kaggleweight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        validation_tree.SetBranchAddress('KaggleSet',  kaggleset_value)
        validation_tree.SetBranchAddress('KaggleWeight',  kaggleweight_value)

        validation_values.append(label_value)
        validation_values.append(weight_value)
        validation_values.append(kaggleset_value)
        validation_values.append(kaggleweight_value)
        #label_value = array.array( 'c',  ['x'])
        #weight_value = array.array( 'f',  [0.0])
    
        # Add label and weight
        #validation_tree.SetBranchAddress('Label',  label_value)
        #validation_tree.SetBranchAddress('Weight',  weight_value)
        #validation_tree.Print()
        validation_tree.Scan('Label')
        validation_tree.Scan('KaggleSet')

        nevt = validation_tree.GetEntries()
        data = []
        actualdata = []
        # iterate over all events. if any entry in the specified indices list is -999 discard that event.
        # otherwise write it to new file. Thats the basic idea
        for evt in xrange(2):#nevt):
            validation_tree.GetEntry(evt)
            print(validation_values)
            data.append(validation_values)
            actualdata.append(extractDataFromRoot(data[-1]))
        print(len(actualdata))
        print(actualdata[0])    