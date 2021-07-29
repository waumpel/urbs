import pandas as pd
import numpy as np
import os
# SCENARIO GENERATORS
# In this script a variety of scenario generator functions are defined to
# facilitate scenario definitions.

def variable_distribution_share(data, cross_scenario_data, transdist_share):
    # save transdist share for different distribution network shares in different scenarios
    data['transdist_share'] = pd.Series([transdist_share])  # defined as series to avoid validation error
    if transdist_share < 1:
        if bool(cross_scenario_data): #&len(cross_scenario_data['PV_cap_shift'].values)==16
            # expand cap-up capacities of PV_utility_rooftop according to the distribution network share to consider equal maximal pv rooftop capacities
            data['process'].loc[pd.IndexSlice[:, :, 'PV_utility_rooftop'], 'cap-up'] = data['process'].loc[pd.IndexSlice[:, :,'PV_utility_rooftop'], 'cap-up'].values \
                                                                                       + (1 - transdist_share) * cross_scenario_data['PV_cap_shift'].values
            # add additional demand on transmission level (BEV, Heat) from cross_scenario dict
            additional_demand_mobility = cross_scenario_data['mobility_transmission_shift']
            additional_demand_heat = cross_scenario_data['heat_transmission_shift']
        else:
            ###load additional demand (BEV, Heat)
            data['process'].loc[pd.IndexSlice[:, :, 'PV_utility_rooftop'], 'cap-up'] += (1 - transdist_share) * np.array([5148, 4800, 21485, 25560, 899, 11952, 2425, 2628, 15529, 29259, 8063, 5726, 2014, 7535, 4354, 4275])
            additional_demand_mobility = pd.read_excel(open(os.path.join(os.getcwd(), 'Input','additional_demand.xlsx'), 'rb'), sheet_name='mobility', index_col = [0, 1])
            additional_demand_heat = pd.read_excel(open(os.path.join(os.getcwd(), 'Input', 'additional_demand.xlsx'), 'rb'),sheet_name='heat', index_col = [0, 1])

            # this values may not be fitting to new scenarios
            cross_scenario_data['predefClusterOrder'] = [2,1,1,1,3,3,1,3,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2]
            cross_scenario_data['predefClusterCenterIndices'] = [29,11,50,5]

        for col in data['demand']:
            if col[0] in list(additional_demand_mobility.columns):
                data['demand'].loc[:, col] += additional_demand_mobility.loc[:, col[0]] * (1 - transdist_share)
            if col[0] in list(additional_demand_heat.columns):
                data['demand'].loc[:, col] += additional_demand_heat.loc[:, col[0]]  * (1 - transdist_share)
    return data, cross_scenario_data

def transdist100(data, cross_scenario_data):
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'].iloc[0] = 1
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 1)
    return data, cross_scenario_data

def transdist75(data, cross_scenario_data):
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'].iloc[0] = 1
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 0.75)
    return data, cross_scenario_data

def transdist66(data, cross_scenario_data):
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'].iloc[0] = 1
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 0.66)
    return data, cross_scenario_data

def transdist50(data, cross_scenario_data):
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'].iloc[0] = 1
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 0.5)
    return data, cross_scenario_data

def transdist33(data, cross_scenario_data):
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'].iloc[0] = 1
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 0.33)
    return data, cross_scenario_data

def transdist25(data, cross_scenario_data):
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'].iloc[0] = 1
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 0.25)
    return data, cross_scenario_data
    
def transmission(data, cross_scenario_data):
    ###set transdist parameter to zero
    data['global_prop'].loc[pd.IndexSlice[:, 'TransDist'], 'value'] = 0
    # variable_distribution_share(data, cross_scenario_data, 0)
    data, cross_scenario_data = variable_distribution_share(data, cross_scenario_data, 0)
    return data, cross_scenario_data





def scenario_base(data):
    # do nothing
    return data

def scenario_stock_prices(data):
    # change stock commodity prices
    co = data['commodity']
    stock_commodities_only = (co.index.get_level_values('Type') == 'Stock')
    co.loc[stock_commodities_only, 'price'] *= 1.5
    return data


def scenario_co2_limit(data):
    # change global CO2 limit
    global_prop = data['global_prop']
    for stf in global_prop.index.levels[0].tolist():
        global_prop.loc[(stf, 'CO2 limit'), 'value'] *= 0.05
    return data


def scenario_co2_tax_mid(data):
    # change CO2 price in Mid
    co = data['commodity']
    for stf in data['global_prop'].index.levels[0].tolist():
        co.loc[(stf, 'Mid', 'CO2', 'Env'), 'price'] = 50
    return data


def scenario_north_process_caps(data):
    # change maximum installable capacity
    pro = data['process']
    for stf in data['global_prop'].index.levels[0].tolist():
        pro.loc[(stf, 'North', 'Hydro plant'), 'cap-up'] *= 0.5
        pro.loc[(stf, 'North', 'Biomass plant'), 'cap-up'] *= 0.25
    return data


def scenario_no_dsm(data):
    # empty the DSM dataframe completely
    data['dsm'] = pd.DataFrame()
    return data


def scenario_all_together(data):
    # combine all other scenarios
    data = scenario_stock_prices(data)
    data = scenario_co2_limit(data)
    data = scenario_north_process_caps(data)
    return data
