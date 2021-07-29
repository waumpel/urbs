from urbs.identify import *
import copy
import math
import numpy as np
import os

def create_transdist_data(data, microgrid_data_initial, cross_scenario_data):
    mode = identify_mode(data)
    # read standard load profile from csv
    loadprofile_BEV = pd.read_csv(os.path.join(os.getcwd(), 'Input', 'Microgrid_types', 'loadprofile_BEV.csv'))
    loadprofile_BEV = loadprofile_BEV.set_index(data['demand'].index).squeeze()
    # prepare storage dicts for the demand shift to the transmission simulation
    mobility_transmission_shift = dict()
    heat_transmission_shift = dict()
    # remember the efficiency at the transmission-distribution interface to consider in demand shift
    transdist_eff = microgrid_data_initial[0]['transmission'].loc[pd.IndexSlice[:,'top_region_dummy',:,'tdi',:], 'eff'].values[0]

    # define lists to be used in loops
    microgrid_set_list = build_set_list(data)
    microgrid_multiplicator_list = build_multiplicator_list(data)

    # process microgrid data for every region and microgrid type
    for set_number, set in enumerate(microgrid_set_list):  # top region microgrid setting
        top_region_name = data['site'].index.get_level_values(1)[set_number]
        for type_nr, quantity_nr in enumerate(set):  # Auflistung des settings
            microgrid_entries = microgrid_data_initial[type_nr]['site'].index.get_level_values(1)
            n = 0
            while n < quantity_nr:
                microgrid_data_input = copy.deepcopy(microgrid_data_initial[type_nr])
                for entry in microgrid_entries:
                    # microgrids are derived from the predefined microgrid types and setting
                    create_microgrid_data(microgrid_data_input, entry, n, top_region_name)
                n += 1
                # scale capacities, commodities, demand, areas and the loadprofile with multiplicator number of the microgrid
                microgrid_data_input, demand_shift = multiplicator_scaling(mode, data, microgrid_data_input,
                                                                           microgrid_multiplicator_list, set_number, type_nr)
                # copy SupIm data from respective state to the microgrid within that state
                copy_SupIm_data(data, microgrid_data_input, top_region_name)
                # shift demand from transmission level to distribution level
                data, mobility_transmission_shift, heat_transmission_shift = shift_demand(data, microgrid_data_input, set_number,
                                                                                          type_nr, demand_shift, loadprofile_BEV,
                                                                                          top_region_name, mobility_transmission_shift,
                                                                                          heat_transmission_shift, transdist_eff)
                # model additional transmission lines for the reactive power
                add_reactive_transmission_lines(microgrid_data_input)
                # add reactive output ratios for ac sites
                add_reactive_output_ratios(microgrid_data_input)
                # concatenate main data with microgrid data with
                concatenate_with_micros(data, microgrid_data_input)
    if data['transdist_share'].values[0] == 1:
        store_additional_demand(cross_scenario_data, mobility_transmission_shift, heat_transmission_shift)
    return data, cross_scenario_data

def build_set_list(data):
    transdist_dict = data['site'].drop(columns=['base-voltage','area'], axis=1).to_dict()
    microgrid_set = transdist_dict['microgrid-setting']
    microgrid_set_list = []
    for item in microgrid_set:
        microgrid_set_list.append(list(map(int, microgrid_set[item].split(','))))
    return microgrid_set_list

def build_multiplicator_list(data):
    transdist_dict = data['site'].drop(columns=['base-voltage','area'], axis=1).to_dict()
    microgrid_multiplicator = transdist_dict['multiplicator']
    microgrid_multiplicator_list = []
    for item in microgrid_multiplicator:
        microgrid_multiplicator_list.append(list(map(int, microgrid_multiplicator[item].split(','))))
    return microgrid_multiplicator_list

# In this function according to the microgrid-setting list and the defined microgrid types, microgrid data are created
def create_microgrid_data(microgrid_data_input, entry, n, top_region_name):
    microgrid_data_input['site'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['commodity'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['process'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['process_commodity'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['demand'].rename(
        columns={entry: entry + str(n + 1) + '_' + top_region_name}, level=0, inplace=True)
    microgrid_data_input['supim'].rename(
        columns={entry: entry + str(n + 1) + '_' + top_region_name}, level=0, inplace=True)
    microgrid_data_input['storage'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['dsm'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['buy_sell_price'].rename(
        columns={entry: entry + str(n + 1) + '_' + top_region_name}, level=0, inplace=True)
    microgrid_data_input['eff_factor'].rename(
        columns={entry: entry + str(n + 1) + '_' + top_region_name}, level=0, inplace=True)
    # for transmission data indexes on two levels must be changed
    microgrid_data_input['transmission'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=1, inplace=True)
    microgrid_data_input['transmission'].rename(
        index={entry: entry + str(n + 1) + '_' + top_region_name}, level=2, inplace=True)
    # add transmission line from microgrids to top level region
    microgrid_data_input['transmission'].rename(
        index={'top_region_dummy': top_region_name}, level=1, inplace=True)
    microgrid_data_input['transmission'].rename(
        index={'top_region_dummy': top_region_name}, level=2, inplace=True)
    return microgrid_data_input

# In this function according to the multiplicator list microgrid types are being scaled
def multiplicator_scaling(mode, data, microgrid_data_input, microgrid_multiplicator_list, set_number, type_nr):
    multi = data['transdist_share'].values[0] * microgrid_multiplicator_list[set_number][type_nr]
    microgrid_data_input['site'].loc[:, 'base-voltage'] *= math.sqrt(multi)
    microgrid_data_input['commodity'].loc[:, 'max':'maxperhour'] *= multi
    microgrid_data_input['process'].loc[:, ['inst-cap', 'cap-lo', 'cap-up', 'cap-block']] *= multi
    microgrid_data_input['transmission'].loc[:, ['inst-cap', 'cap-lo', 'cap-up', 'tra-block']] *= multi
    microgrid_data_input['storage'].loc[:, ['inst-cap-c', 'cap-lo-c', 'cap-up-c', 'inst-cap-p', 'cap-lo-p',
                                            'cap-up-p', 'c-block', 'p-block']] *= multi
    microgrid_data_input['dsm'].loc[:, 'cap-max-do':'cap-max-up'] *= multi
    # if tsam activated postpone demand scaling to reduce number of tsam input timeseries, but still pass demand shift
    if mode['tsam'] == True:
        demand_shift = microgrid_data_input['demand'] * multi
    # otherwise also scale demand data
    if mode['tsam'] == False:
        microgrid_data_input['demand'] *= multi
        demand_shift = microgrid_data_input['demand']
    return microgrid_data_input, demand_shift

def shift_demand(data, microgrid_data_input, set_number, type_nr, demand_shift, loadprofile_BEV, top_region_name,
                 mobility_transmission_shift, heat_transmission_shift, transdist_eff):
    ### subtract private electricity demand at distribution level (increased by tdi efficiency) from transmission level considering line losses
    data['demand'].iloc[:, set_number] -= demand_shift.loc[:, pd.IndexSlice[:, 'electricity']].sum(axis=1) / transdist_eff
    if data['transdist_share'].values[0] == 1:
        ### store scaled mobility load profile for each state only for full transdist model to add share of demand in scenarios
        mobility_transmission_shift[(top_region_name, type_nr)] = loadprofile_BEV * demand_shift.loc[:, pd.IndexSlice[:, 'mobility']].sum().sum() / transdist_eff
        COP_ts = microgrid_data_input['eff_factor'].loc[:, pd.IndexSlice[:, 'heatpump_air']].iloc[:,0].squeeze()
        heat_transmission_shift[(top_region_name, type_nr)] = demand_shift.loc[:, pd.IndexSlice[:, 'heat']].sum(axis=1).divide(COP_ts).fillna(0) / transdist_eff
    return data, mobility_transmission_shift, heat_transmission_shift

def copy_SupIm_data(data, microgrid_data_input, top_region_name):
    for col in microgrid_data_input['supim'].columns:
        microgrid_data_input['supim'].loc[:, col] = data['supim'].loc[:, (top_region_name, col[1])]
    return microgrid_data_input

# In this function according to predefined resistances on lines reactive power flows are enabled by modeling new lines
def add_reactive_transmission_lines(microgrid_data_input):
    # copy transmission lines with resistance to model transmission lines for reactive power flows
    reactive_transmission_lines = microgrid_data_input['transmission'][microgrid_data_input['transmission'].loc[:, 'resistance'] > 0]
    reactive_transmission_lines = reactive_transmission_lines.copy(deep = True)
    reactive_transmission_lines.rename(index={'electricity': 'electricity-reactive'}, level=4, inplace=True)
    # set costs to zero
    reactive_transmission_lines.loc[:, 'inv-cost':'var-cost'] *= 0
    # scale transmission line capacities with predefined Q/P-ratio
    for idx, entry in reactive_transmission_lines.iterrows():
        reactive_transmission_lines.loc[idx, ['inst-cap', 'cap-lo', 'cap-up']] = entry['inst-cap':'cap-up'] * entry['cap-Q/P-ratio']
    microgrid_data_input['transmission'] = pd.concat([microgrid_data_input['transmission'], reactive_transmission_lines], sort=True)
    return microgrid_data_input

### In this function according to predefined power factors for processes, reactive power outputs are implemented as commodity
def add_reactive_output_ratios(microgrid_data_input):
    pro_Q = microgrid_data_input['process'][microgrid_data_input['process'].loc[:, 'pf-min'] > 0]
    ratios_elec = microgrid_data_input['process_commodity'].loc[pd.IndexSlice[:, :, 'electricity', 'Out'], :]
    for process_idx, process in pro_Q.iterrows():
        for ratio_P_idx, ratio_P in ratios_elec.iterrows():
            if process_idx[2] == ratio_P_idx[1]:
                ratio_Q = ratios_elec.loc[pd.IndexSlice[:, ratio_P_idx[1], 'electricity', 'Out'], :].copy(deep = True)
                ratio_Q.rename(index={'electricity': 'electricity-reactive'}, level=2, inplace=True)
                microgrid_data_input['process_commodity'] = microgrid_data_input['process_commodity'].append(ratio_Q)
                microgrid_data_input['process_commodity'] = microgrid_data_input['process_commodity']\
                [~microgrid_data_input['process_commodity'].index.duplicated(keep='first')]
    return microgrid_data_input

### In this function the main data and the microgrid data are merged
def concatenate_with_micros(data, microgrid_data):
    data['site'] = pd.concat([data['site'], microgrid_data['site']], sort=True)
    data['commodity'] = pd.concat([data['commodity'], microgrid_data['commodity']],sort=True)
    data['process'] = pd.concat([data['process'], microgrid_data['process']],sort=True)
    data['process_commodity'] = pd.concat([data['process_commodity'], microgrid_data['process_commodity']],sort=True)
    ### delete duplicated process commodities (for different ratios from different systems, the processes need adapted names)
    data['process_commodity'] = data['process_commodity'][~data['process_commodity'].index.duplicated(keep='first')]
    data['demand'] = pd.concat([data['demand'], microgrid_data['demand']], axis=1,sort=True)
    data['supim'] = pd.concat([data['supim'], microgrid_data['supim']], axis=1,sort=True)
    data['transmission'] = pd.concat([data['transmission'], microgrid_data['transmission']],sort=True)
    data['storage'] = pd.concat([data['storage'], microgrid_data['storage']],sort=True)
    data['dsm'] = pd.concat([data['dsm'], microgrid_data['dsm']],sort=True)
    data['buy_sell_price'] = pd.concat([data['buy_sell_price'], microgrid_data['buy_sell_price']], axis=1,sort=True)
    data['eff_factor'] = pd.concat([data['eff_factor'], microgrid_data['eff_factor']], axis=1,sort=True)
    return data

def store_additional_demand(cross_scenario_data, mobility_transmission_shift, heat_transmission_shift):
    ###transform dicts into dataframe and summarize timeseries for regions
    mobility_transmission_shift = pd.DataFrame.from_dict(mobility_transmission_shift).sum(level=0, axis=1)
    heat_transmission_shift = pd.DataFrame.from_dict(heat_transmission_shift).sum(level=0, axis=1)
    heat_transmission_shift.index = pd.MultiIndex.from_tuples(mobility_transmission_shift.index)
    ###write data into an excel file
    with pd.ExcelWriter(os.path.join(os.path.join(os.getcwd(), 'Input', 'additional_demand.xlsx'))) as writer:
        mobility_transmission_shift.to_excel(writer, 'mobility')
        heat_transmission_shift.to_excel(writer, 'heat')
    ###save cross scenario data in dict
    cross_scenario_data['mobility_transmission_shift'] = mobility_transmission_shift
    cross_scenario_data['heat_transmission_shift'] = heat_transmission_shift

    return cross_scenario_data