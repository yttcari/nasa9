import sys
import math
import subprocess
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import polars as pl
import argparse as ap
import seaborn as sns
import Constants as c
import config as cfg
import Aux as aux

##### SETTINGS ######
overlay_measurements = True
measurement = "JupiterUpperAtm"
measurement_label = "Galileo Measurement"
pressure_exists = True
altitude_exists = True
colours = [
    "#FF6B6B",  # Coral pink (vibrant but soft)
    "#4ECDC4",  # Tiffany blue (fresh turquoise)
    "#2ECC71",  # Sky blue (bright but not harsh)
    "#A05195",  # Dusty purple (rich, works for colorblind)
    "#FFA600",  # Golden yellow (high contrast)
    "#82B446",  # Lime green (poppy but readable)
    "#D35FB7",  # Raspberry pink (fun yet distinct)
    "#003F5C",  # Deep navy (for contrast)
]
CB_color_cycle = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # violet
    "#e7298a",  # magenta
    "#66a61e",  # green
    "#e6ab02",  # mustard
    "#a6761d",  # brown
    "#666666",  # gray
    "#1f78b4",  # blue
]
# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
#                   '#f781bf', '#a65628', '#984ea3',
#                   '#999999', '#e41a1c', '#dede00']
cmap = plt.get_cmap("Dark2")
colours_10 = cmap(np.linspace(0, 1, 10))
sns.set_theme(palette=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])

plt.rcParams['axes.grid'] = False

def extract_and_plot_pt(args):
    # Handle input
    single = False
    print("Running plots...")
    if isinstance(args.ext_in, str):
        file_ext_list = [args.ext_in]
        single = True
    elif isinstance(args.ext_in, list):
        file_ext_list = args.ext_in
        if len(file_ext_list) == 1:
            single = True
    else:
        raise TypeError("file_ext_in must be a string or a list of strings")

    # Extract all the profiles uwu
    thermal_dfs = []
    for ext_in in file_ext_list:
        thermal_dfs.append(extract_profile(ext_in))

    plot_properties(thermal_dfs, args.ext_in, args.ext_out, args.t, single)


def extract_and_plot_chem(args, species):
    # Handle input
    single = False
    print("Running plots (chem)...")
    if isinstance(args.ext_in, str):
        file_ext_list = [args.ext_in]
        single = True
    elif isinstance(args.ext_in, list):
        file_ext_list = args.ext_in
        if len(file_ext_list) == 1:
            single = True
    else:
        raise TypeError("file_ext_in must be a string or a list of strings")

    chem_dfs = []
    thermal_dfs = []
    for ext_in in file_ext_list:
        chem_dfs.append(extract_species(ext_in))
        thermal_dfs.append(extract_profile(ext_in))
    plot_chem(thermal_dfs, chem_dfs, args.ext_out, args.t, species, single)


def extract_measurement(m_name):
    filename = "TestData/Measurements/" + m_name + ".dat"
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    data_matrix = np.array([col.split() for col in content])
    data_matrix = data_matrix.astype(float)
    thermal_df = pl.DataFrame(data=data_matrix)

    return thermal_df


def kompot_to_full_planet_profile(ext):
    """
    TODO
    Takes the ext of a kompot run to turn into a full planet profile.
    """
    return 0


def get_project():
    """
    TODO
    Gets the current project from kompot to use for finding files
    """
    return 0

def compare_heating_cooling(therms, labels, ext):
    
    fig1, ax1= plt.subplots(figsize = (8,6))
    fig2, ax2= plt.subplots(figsize = (8,6))
    fig3, ax3= plt.subplots(figsize = (8,6))
    fig4, ax4= plt.subplots(figsize = (8,6))
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    
    for i, therm in enumerate(therms):
        P = therm["Pressure"]
        A = therm["Altitude"]
        print(i)
        conduction = np.sign(therm["Conduction heating function"])*np.log10(np.abs(np.divide(therm["Conduction heating function"], therm["Mass density"]) + 1e-20))
        ax1.plot(np.divide(therm["Total heating rate"], therm["Mass density"]), P, c=CB_color_cycle[i], label = labels[i])
        ax2.plot(np.divide(therm["Total cooling rate"], therm["Mass density"]), P, c=CB_color_cycle[i], label = labels[i])
        
        ax3.plot(np.log10(np.divide(therm["Total heating rate"], therm["Mass density"])), P, label="Total heating", c=CB_color_cycle[1], linestyle = linestyles[i])
        ax3.plot(-np.log10(np.divide(therm["Total cooling rate"], therm["Mass density"])), P, label="Total cooling", c=CB_color_cycle[2], linestyle = linestyles[i])
        ax3.plot(conduction, P, label="Conduction", c=CB_color_cycle[0], linestyle = linestyles[i])
        ax4.plot(np.log10(np.divide(therm["Total heating rate"], therm["Mass density"])), A, label="Total heating", c=CB_color_cycle[1], linestyle = linestyles[i])
        ax4.plot(np.log10(np.divide(therm["Total cooling rate"], therm["Mass density"])), A, label="Total cooling", c=CB_color_cycle[2], linestyle = linestyles[i])
        ax4.plot(np.log10(np.divide(therm["Conduction heating function"], therm["Mass density"])), A, label="Conduction", c=CB_color_cycle[0], linestyle = linestyles[i])      
    ax1.invert_yaxis()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylim(1e-3, 1e-10)
    ax1.legend(fontsize = 'large')
    ax1.set_xlim(1e-1, 1e10)
    ax1.set_xlabel("Heating [erg $s^{-1}$ $g^{-1}$]", fontsize = 18)
    ax1.set_ylabel("Pressure [bar]", fontsize = 18)
    
    ax2.invert_yaxis()
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylim(1e-3, 1e-10)
    ax2.legend(fontsize = 'large')
    ax2.set_xlim(1e-1, 1e10)
    ax2.set_xlabel("Cooling [erg $s^{-1}$ $g^{-1}$]", fontsize = 18)
    ax2.set_ylabel("Pressure [bar]", fontsize = 18)
    
    ax3.invert_yaxis()
    ax3.set_yscale("log")
    ax3.set_ylim(1e-3, 1e-10)
    ax3.set_xlim(-5, 5)
    legend_el = [Line2D([0],[0], color=CB_color_cycle[1], label = 'Total heating'), Line2D([0],[0], color=CB_color_cycle[0], label = 'Conduction'), 
                 Line2D([0],[0], color=CB_color_cycle[2], label = 'Total cooling'), Line2D([0],[0],linestyle = linestyles[0], color='black', label = labels[0]), 
                 Line2D([0],[0], linestyle= linestyles[1],color='black', label = labels[1]),
                 Line2D([0],[0], linestyle = linestyles[2],color='black', label = labels[2]), Line2D([0],[0], linestyle= linestyles[3], color='black', label = labels[3])]
    ax3.legend(handles=legend_el, loc = (0.48,0.46))#, loc = (0.48,0.46))
    ax3.set_xlabel("Log$_{10}$ Heating/Cooling [erg $s^{-1}$ $g^{-1}$]", fontsize = 16)
    ax3.set_ylabel("Pressure [bar]", fontsize = 16)
    ax3.tick_params(labelsize = 14)
    ax3.grid(alpha = 0)
    fig3.savefig("Plots/HC_COMP_LOGG" + ext + ".pdf")
    fig1.savefig("Plots/H_COMP_" + ext + ".pdf")
    fig2.savefig("Plots/C_COMP_" + ext + ".pdf")
    
    #ax4.invert_yaxis()
    #ax4.set_yscale("log")
    ax4.set_ylim(150, 1300)
    #ax3.set_xlim(-5, 5)
    legend_el = [Line2D([0],[0], color=CB_color_cycle[1], label = 'Total heating'), Line2D([0],[0], color=CB_color_cycle[0], label = 'Conduction'), 
                 Line2D([0],[0], color=CB_color_cycle[2], label = 'Total cooling'), Line2D([0],[0],linestyle = linestyles[0], color='black', label = labels[0]), 
                 Line2D([0],[0], linestyle= linestyles[1],color='black', label = labels[1]),
                 Line2D([0],[0], linestyle = linestyles[2],color='black', label = labels[2]), Line2D([0],[0], linestyle= linestyles[3], color='black', label = labels[3])]
    ax4.legend(handles=legend_el)#, loc = (0.48,0.46))
    ax4.set_xlabel("Log$_{10}$ Heating/Cooling [erg $s^{-1}$ $g^{-1}$]", fontsize = 18)
    ax4.set_ylabel("Altitude [km]", fontsize = 18)
    ax4.grid(alpha = 0)
    fig4.savefig("Plots/HC_COMP_LOGG_ALT" + ext + ".pdf")
    
    return ax1, ax2, fig1, fig2

def plot_heating_cooling(
    ext, measurements=[], measurement_labels=[], measurements_inc=False
):
    """
    Plots all the heating and cooling functions.
    Measurements should be structured as [[heating x, heating y],[cooling x, cooling y]]
    """

    profile = extract_profile(ext)
    P = profile["Pressure"]
    A = profile["Altitude"]
    integrated_rates = []
    fig1, p1 = plt.subplots(figsize = (8,6))
    fig2, p2 = plt.subplots(figsize = (8,6))
    fig3, p3 = plt.subplots(figsize = (8,6))
    fig4, p4 = plt.subplots(figsize = (8,6))
    
    
    conduction = np.sign(profile["Conduction heating function"])*np.log10(np.abs(np.divide(profile["Conduction heating function"], profile["Mass density"]) + 1e-20))
    p1.plot(np.log10(np.divide(profile["Total heating rate"], profile["Mass density"])), P, label="Total heating", c=CB_color_cycle[1])
    #p1.plot(-np.log10(np.divide(profile["Total cooling rate"], profile["Mass density"])), P, label="Total cooling", c=CB_color_cycle[0])
    #p1.plot(conduction, P, label="Conduction", c=CB_color_cycle[2])
    p1.invert_yaxis()
    p1.set_yscale("log")
    p1.set_ylim(1e-3, 1e-12)
    p1.set_xlim(-10, 10)
    p1.legend(loc = 7)
    p1.set_xlabel("Log$_{10}$ Heating/Cooling [erg $s^{-1}$ $g^{-1}$]", fontsize = 18)
    p1.set_ylabel("Pressure [bar]", fontsize = 18)
    fig1.savefig("Plots/HC_comp_" + ext + ".pdf")
    lw = 3
    if "XUV heating" in profile:
        p2.plot(np.divide(profile["XUV heating"], profile["Mass density"]), P,
                label="XUV", c=CB_color_cycle[3], linewidth=lw)

    if "Chemical heating" in profile:
        p2.plot(np.divide(profile["Chemical heating"], profile["Mass density"]), P,
                label="Chemical", c=CB_color_cycle[4], linewidth=lw)

    if "Joule heating" in profile:
        p2.plot(np.divide(profile["Joule heating"], profile["Mass density"]), P,
                label="Joule", c=CB_color_cycle[8], linewidth=lw)

    if "Photoelectron heating" in profile:
        p2.plot(np.divide(profile["Photoelectron heating"], profile["Mass density"]), P,
                label="Photoelectron", c=CB_color_cycle[6], linewidth=lw)

    if "IR heating" in profile:
        p2.plot(np.divide(profile["IR heating"], profile["Mass density"]), P,
                label="IR", c=CB_color_cycle[7], linewidth=lw)

    if "Total heating rate" in profile:
        p2.plot(np.divide(profile["Total heating rate"], profile["Mass density"]), P,
                label="Total heating", linestyle="--", c=CB_color_cycle[1], linewidth=lw)

    p2.invert_yaxis()
    p2.set_yscale("log")
    p2.set_xscale("log")
    p2.set_ylim(1e-3, 1e-12)
    p2.legend(fontsize='medium')
    p2.set_xlim(1e-10, 1e10)
    p2.grid(alpha = 0)
    p2.set_xlabel("Heating [erg $s^{-1}$ $g^{-1}$]", fontsize=18)
    p2.set_ylabel("Pressure [bar]", fontsize=18)
    p2.tick_params(labelsize = 14)
    p2.grid(alpha = 0)
    fig2.savefig("Plots/H_" + ext + ".pdf")


    # Cooling terms
    if "CO2 cooling" in profile:
        p3.plot(np.divide(profile["CO2 cooling"], profile["Mass density"]), P,
                label="CO$_2$", c=CB_color_cycle[2], linewidth=lw)

    if "NO cooling" in profile:
        p3.plot(np.divide(profile["NO cooling"], profile["Mass density"]), P,
                label="NO", c=CB_color_cycle[4], linewidth=lw)

    if "O cooling" in profile:
        p3.plot(np.divide(profile["O cooling"], profile["Mass density"]), P,
                label="O", c=CB_color_cycle[6], linewidth=lw)

    if "H3+ cooling" in profile:
        p3.plot(np.divide(profile["H3+ cooling"], profile["Mass density"]), P,
                label="H$_3^+$", c=CB_color_cycle[5], linewidth=lw)

    if "H2O cooling" in profile:
        p3.plot(np.divide(profile["H2O cooling"], profile["Mass density"]), P,
                label="H$_2$O", c=CB_color_cycle[7], linewidth=lw)

    if "Gravitational cooling" in profile:
        p3.plot(np.divide(profile["Gravitational cooling"], profile["Mass density"]), P,
                label="Grav.", c=CB_color_cycle[8], linewidth=lw)

    if "CH4 cooling" in profile:
        p3.plot(np.divide(profile["CH4 cooling"], profile["Mass density"]), P,
                label="CH$_4$", c=CB_color_cycle[3], linewidth=lw)
    if "CO cooling" in profile:
        p3.plot(np.divide(profile["CO cooling"], profile["Mass density"]), P,
                label="CO", c=CB_color_cycle[1], linewidth=lw)

    if "Total cooling rate" in profile:
        p3.plot(np.divide(profile["Total cooling rate"], profile["Mass density"]), P,
                label="Total cooling", linestyle="--", c=CB_color_cycle[0], linewidth=lw)

    p3.set_yscale("log")
    p3.set_xscale("log")
    p3.invert_yaxis()
    p3.set_ylim(1e-3, 1e-12)
    p3.set_xlim(1e-10, 1e10)
    p3.legend(fontsize = 'medium')
    p3.grid(alpha = 0)
    p3.set_xlabel("Cooling [erg $s^{-1}$ $g^{-1}$]", fontsize = 18)
    p3.set_ylabel("Pressure [bar]", fontsize = 18)
    p3.grid(alpha = 0)
    p3.tick_params(labelsize = 14)
    fig3.savefig("Plots/C_" + ext + ".pdf")
    
    if measurements_inc:
        for i,label in enumerate(measurement_labels):
            p4.plot(measurements[i][0], measurements[i][1], label = label, linestyle = "--")
        p4.plot(profile["Total heating rate"], P, label="Total heating", c="maroon")
        p4.plot(profile["Total cooling rate"], P, label="Total cooling", c="navy")

    p4.invert_yaxis()
    p4.set_yscale("log")
    p4.set_xscale("log")
    p4.set_ylim(1e-3, 1e-12)
    #p4.set_xlim(1e-7, 1e6)
    p4.legend(loc="upper left")
    p4.set_xlabel("Heating/Cooling [erg $s^{-1}$ $cm^{-3}$]", fontsize = 18)
    p4.set_ylabel("Pressure [bar]", fontsize = 18)
    fig4.savefig("Plots/TEST" + ext)

    ############# Height integrated stuff + table ##################

    # Cooling terms
    if "CO2 cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["CO2 cooling"]), "CO2 cooling"])

    if "NO cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["NO cooling"]), "NO cooling"])

    if "O cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["O cooling"]), "O cooling"])

    if "H3+ cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["H3+ cooling"]), "H3+ cooling"])

    if "H2O cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["H2O cooling"]), "H2O cooling"])

    # if "Gravitational cooling" in profile:
    #     integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["Gravitational cooling"]), "Gravitational cooling"])

    if "CH4 cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["CH4 cooling"]), "CH4 cooling"])
        
    if "HCN cooling" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["HCN cooling"]), "HCN cooling"])

    if "Total cooling rate" in profile:
        integrated_rates.append([-aux.integrate_vol_rate(profile["Altitude"], profile["Total cooling rate"]), "Total cooling rate"])
        
    if "XUV heating" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["XUV heating"]), "XUV heating"])

    if "Chemical heating" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["Chemical heating"]), "Chemical heating"])

    if "Joule heating" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["Joule heating"]), "Joule heating"])

    if "Photoelectron heating" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["Photoelectron heating"]), "Photoelectron heating"])

    if "IR heating" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["IR heating"]), "IR heating"])

    if "Total heating rate" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["Total heating rate"]), "Total heating rate"])
        
    if "Conduction heating function" in profile:
        integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"], profile["Conduction heating function"]), "Conduction heating function"])
        
    total_mass = sum(profile["Mass density"])
    net_term = profile['Total heating rate'][0:] - profile['Total cooling rate'][0:] + profile['Conduction heating function'][0:]
    integrated_rates.append([aux.integrate_vol_rate(profile["Altitude"][0:], net_term), "Net integration"])
    boundary_L = np.abs(profile["Conduction heating function"][1:5])
    boundary_L_A = profile["Altitude"][1:5]
    boundary_U = np.abs(profile["Conduction heating function"][-3:])
    boundary_U_A = profile["Altitude"][-3:]
    print(boundary_U)
    boundary_L_int = aux.integrate_vol_rate(boundary_L_A, boundary_L)
    boundary_U_int = aux.integrate_vol_rate(boundary_U_A, boundary_U)
    
    print(f'BOUNDARY CONV = {boundary_L_int}, UL = {boundary_U_int}')
    totals = 0
    totals -= 47308.51624855582
    print(ext)
    for item in integrated_rates:
        print(f'Total energy: {item[0]} TW from {item[1]}')
        # 863 - 1080 = 220 TW
        if(item[1] != 'Gravitational cooling' or  item[1] != 'Total heating rate' or item[1] != 'Total cooling rate'):
            totals += item[0]
    
    #print(f'Total conduction is {} TW')
        
def full_planet_profile(P, T, Z, N, mix_ratios, Kzz):
    """
    TODO
    Plots the full planet profile as a collage and saves as planet_profile_ext.pdf in Plots
    Plot 1: Temp vs Pressure
    Plot 2: Temp vs Altitude
    Plot 3: Number density vs Pressure
    Plot 4: Number density vs Altitude
    Plot 5: Mixing ratios vs Pressure
    Plot 6: Mixing ratios vs Altitude
    Plot 7: Eddy mixing coefficients vs Pressure
    Plot 8: To be filled
    Plot 9: To be filled
    """

    fig, axs = plt.subplots(3, 3)

    axs[0][0].set_xlabel("Temperature [K]")
    axs[0][0].set_ylabel("Pressure [bar]")

    axs[0][0].set_xlabel("Temperature [K]")
    axs[0][0].set_ylabel("Altitude [km]")

    axs[0][0].set_xlabel("Number density [$cm^{-3}$]")
    axs[0][0].set_ylabel("Pressure [bar]")

    axs[0][0].set_xlabel("Number density [$cm^{-3}$]")
    axs[0][0].set_ylabel("Altitude [km]")

    axs[0][0].set_xlabel("Mixing ratio")
    axs[0][0].set_ylabel("Pressure [bar]")

    axs[0][0].set_xlabel("Mixing ratio")
    axs[0][0].set_ylabel("Altitude [km]")

    axs[0][0].set_xlabel("Temperature [K]")
    axs[0][0].set_ylabel("Pressure [bar]")

    axs[0][0].set_xlabel("Temperature [K]")
    axs[0][0].set_ylabel("Pressure [bar]")

    axs[0][0].set_xlabel("Temperature [K]")
    axs[0][0].set_ylabel("Pressure [bar]")

def extract_dmol(ext):
    """Takes the thermalProperties and returns the properties in a dataframe."""

    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/Output/Dmol_out.dat"
    df = pd.read_csv(filename, delimiter=' ', header=0 )
    return df

def extract_profile(ext):
    """Takes the thermalProperties and returns the properties in a dataframe."""

    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/FinalState/ThermalProperties.dat"
    print(filename)
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    line1 = content[0].split("=")
    nHeader = int(line1[1]) +2
    line2 = content[1].split("=")
    nrGrid = int(line2[1])
    
    columns_names = []
    # loop over every line in header
    for line in content[0:nHeader]:
        if ":" in line:
            line = line.split(":")[1].strip().split("(")[0].rstrip()
            columns_names.append(line)
    data_matrix = np.array([col.split() for col in content[nHeader : nrGrid + nHeader]])
    columns_names = [(name, pl.Float32) for name in columns_names]
    thermal_df = pl.DataFrame(data=data_matrix, schema=columns_names)

    return thermal_df


def extract_profile_sud(ext):
    """Takes the thermalProperties and returns the properties in a dataframe."""

    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/FinalState/ThermalProperties.dat"
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    line1 = content[0].split("=")
    nHeader = int(line1[1]) +4
    line2 = content[1].split("=")
    nrGrid = int(line2[1])
    print(filename)
    columns_names = []
    # loop over every line in header
    for line in content[0:nHeader]:
        if ":" in line:
            line = line.split(":")[1].strip().split("(")[0].rstrip()
            columns_names.append(line)
    data_matrix = np.array([col.split() for col in content[nHeader : nrGrid + nHeader]])
    columns_names = [(name, pl.Float32) for name in columns_names]
    thermal_df = pl.DataFrame(data=data_matrix, schema=columns_names)

    return thermal_df

def extract_species(ext):
    """
    Reads the species_densities file from kompot paths finalstate. Returns
    a dataframe with column names as index.
    """
    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/FinalState/species_densities.dat"

    with open(filename, "r") as f:
        content = [line.strip() for line in f.readlines()]

    header_line = content[2]
    column_names = []
    for word in header_line.split():
        if "_(" in word:
            column_names.append(word.split("_")[0])
        else:
            column_names.append(word)

    data_lines = content[3:]
    data_matrix = np.array([line.split() for line in data_lines])
    columns_names = [(name, pl.Float32) for name in column_names]
    df = pl.DataFrame(data=data_matrix, schema=columns_names)
    df = df.with_columns(
        pl.sum_horizontal([pl.col(c) for c in column_names[4:]]).alias("Number density")
    )
    return df


def plot_PT_w_meas_arr(
    thermal_dfs,
    label_mod,
    ext,
    title,
    measurements_points,
    measurement_labels,
    error_bars=False,
    single=False,
    meas=False,
    add_altitude=False
):
    """
    Plots the PT diagram(s) from the extracted dataframe(s)
    Measurement needs to have structure [[Measurement xs, measurement ys], [..]...] and error bars the same structure. Error bars need to be symmetric.
    """

    # Create the figure and axis explicitly
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.autoscale()

    # Plot PT profiles
    if single:
        ax.plot(
            thermal_dfs[0]["Total temperature"],
            thermal_dfs[0]["Pressure"],
            label=label_mod[0],
        )
        
    else:
        for i, thermal_df in enumerate(thermal_dfs):
            ax.plot(
                thermal_df["Total temperature"],
                thermal_df["Pressure"],
                label=label_mod[i],
                c=CB_color_cycle[i]
            )

    # Measurements and error bars
    if error_bars and meas:
        for i, point in enumerate(measurements_points):
            ax.errorbar(
                measurements_points[i][1],
                measurements_points[i][0],
                measurements_points[i][2],
                measurements_points[i][3],
                label=measurement_labels[i],
                marker="o",
            )
    elif meas and not error_bars:
        for i, point in enumerate(measurements_points):
            ax.plot(
                measurements_points[i][0],
                measurements_points[i][1],
                label=measurement_labels[i],
                linestyle=":",
            )

    # Labels, scales, etc.
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Temperature [K]", fontsize = 16)
    ax.set_ylabel("Pressure [bar]", fontsize = 16)
    ax.set_yscale("log")
    #ax.set_xlim(0, 1500)
    ax.set_ylim(1e-15, 1e-3)
    ax.grid(alpha = 0)
    ax.invert_yaxis()
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Altitude (km)', color='red')
    # ax2.tick_params(axis='y', labelcolor='red')
    t = thermal_dfs[0]
    ind = min(range(len(t['Pressure'])), key=lambda i: abs(t['Pressure'][i]-1e-10))
    # ax2.set_ylim(t['Altitude'].min(), t['Altitude'][ind]) 

    # Save the plot
    fig.savefig(f"Plots/PTsinglemeas_{ext}.pdf", dpi=300, bbox_inches="tight")

    # Return both figure and axis so you can continue plotting externally
    return fig, ax

def plot_AT_w_meas_arr(
    thermal_dfs,
    label_mod,
    ext,
    title,
    measurements_points,
    measurement_labels,
    error_bars=False,
    single=False,
    meas=False
):
    """
    Plots the PT diagram(s) from the extracted dataframe(s)
    Measurement needs to have structure [[Measurement xs, measurement ys], [..]...] and error bars the same structure. Error bars need to be symmetric.
    """

    plt.figure(figsize=(10, 7))
    plt.autoscale()
    if single:
        plt.plot(
            thermal_dfs[0]["Total temperature"],
            thermal_dfs[0]["Altitude"],
            label=label_mod[0],
        )
    else:
        for i, thermal_df in enumerate(thermal_dfs):
            plt.plot(
                thermal_df["Total temperature"],
                thermal_df["Altitude"],
                label=label_mod[i],
                #linestyle="--",
            )

    if error_bars & meas:
        for i, point in enumerate(measurements_points):
            plt.errorbar(
                measurements_points[i][1],
                measurements_points[i][0],
                measurements_points[i][2],
                measurements_points[i][3],
                label=measurement_labels[i],
                marker="o",
            )
    elif meas == True & error_bars == False:
      for i, point in enumerate(measurements_points):
            plt.plot(
                measurements_points[i][0],
                measurements_points[i][1],
                label=measurement_labels[i],
                linestyle=":",
            )

    plt.legend()
    plt.title(title)
    plt.xlabel("Temperature [K]")
    plt.ylabel("Altitude [km]")
    plt.xlim(100,1100)
    plt.ylim(150, 1200)
    plt.savefig("Plots/ATsinglemeas_" + ext + ".pdf")

    return plt

def plot_thermal_gradients_single(thermal, dtdz, dtdp):
    """
    Plots the thermal gradients lmao
    """
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_xlabel("$\\frac{dT}{dP}$")
    ax1.set_ylabel("Pressure [bar]")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.invert_yaxis()
    ax1.plot(dtdp, thermal["Pressure"])
    
    ax2.set_xlabel("$\\frac{dT}{dZ}$")
    ax2.set_ylabel("Altitude [km]")
    ax2.set_xscale("log")
    ax2.plot(dtdz, thermal["Altitude"])
    
    fig.savefig("Plots/gradtest.pdf")
    

def extract_peder(ext):
    
    filename = cfg.kompot_path +"Projects/" + cfg.current_project  + "/" + ext + "/Output/SigmaP_out.dat"
    peder = pd.read_csv(filename, skiprows=1, delimiter=' ', dtype=pd.Float64Dtype)
    peder = peder.columns.tolist()
    peder = [float(i) for i in peder]
    
    return peder
    
    
def plot_PT_w_meas_pts(
    thermal_dfs,
    label_mod,
    ext,
    title,
    measurements_points,
    measurement_labels,
    single=False,
):
    """
    Plots the PT diagram(s) from the extracted dataframe(s)
    Measurement points need to have structure (pressure, temp, pressure error bar, temp error bar)
    """

    plt.figure(figsize=(10, 7))
    plt.autoscale()
    if single:
        plt.plot(
            thermal_dfs[0]["Total temperature"],
            thermal_dfs[0]["Pressure"],
            label=label_mod[0],
        )
    else:
        for i, thermal_df in enumerate(thermal_dfs):
            plt.plot(
                thermal_df["Total temperature"],
                thermal_df["Pressure"],
                label=label_mod[i],
                linestyle="--",
            )

    for i, point in enumerate(measurements_points):
        plt.errorbar(
            point[1],
            point[0],
            point[2],
            point[3],
            label=measurement_labels[i],
            marker="o",
        )
    if overlay_measurements & pressure_exists:
        measurement_df = extract_measurement(measurement)
        # Correct for millibar vs bar with 1e3 factor
        plt.plot(
            measurement_df["column_3"],
            measurement_df["column_2"] / 1e3,
            label=measurement_label,
            linestyle=":",
        )

    plt.legend()
    plt.title(title)
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [bar]")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.savefig("Plots/PTsinglemeas_" + ext + ".pdf")

    return plt


def plot_properties(thermal_dfs, labels, ext, title, single=False, km_offset=0):
    """Plots the PT diagram(s) from the extracted dataframe(s)"""

    ### PLOT Press - Temp
    plt.figure()
    plt.autoscale()
    if single:
        plt.plot(
            thermal_dfs[0]["Total temperature"],
            thermal_dfs[0]["Pressure"],
            label=labels,
        )
    else:
        for i, thermal_df in enumerate(thermal_dfs):
            plt.plot(
                thermal_df["Total temperature"],
                thermal_df["Pressure"],
                label=labels[i],
                c=colours_10[i],
            )

    if overlay_measurements & pressure_exists:
        measurement_df = extract_measurement(measurement)
        # Correct for millibar vs bar with 1e3 factor
        plt.plot(
            measurement_df["column_3"],
            measurement_df["column_2"] / 1e3,
            label=measurement_label,
            c="maroon",
            linestyle="--",
        )

    plt.legend()
    plt.title(title)
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [bar]")
    plt.yscale("log")
    plt.xlim(0, 800)
    plt.gca().invert_yaxis()
    plt.savefig("Plots/PTdiag_" + ext + ".pdf")
    print(f"Figure saved as PTdiag_{ext}.pdf in the Plots folder")

    ### PLOT Alt - Temp
    plt.figure()
    plt.autoscale()
    if single:
        plt.plot(
            thermal_dfs[0]["Total temperature"], thermal_dfs[0]["Altitude"] - km_offset
        )
    else:
        for i, thermal_df in enumerate(thermal_dfs):
            plt.plot(
                thermal_df["Total temperature"],
                thermal_df["Altitude"] - km_offset,
                label=labels[i],
            )

    if overlay_measurements & altitude_exists:
        measurement_df = extract_measurement(measurement)
        # Correct for millibar vs bar with 1e3 factor
        plt.plot(
            measurement_df["column_3"],
            measurement_df["column_1"],
            label=measurement_label,
        )

    plt.legend()
    plt.title(title)
    plt.xlabel("Temperature [K]")
    plt.ylabel("Altitude [km]")
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.savefig("Plots/ATdiag_" + ext + ".pdf")
    print(f"Figure saved as ATdiag_{ext}.pdf in the Plots folder")


def plot_chem_with_measurements_mix(
    thermal_dfs,
    chem_dfs,
    ext,
    labels,
    title,
    species,
    measurements_x,
    measurements_y,
    measurement_labels,
    single=False,
    error_bars = [],
    mol_name_legend = True
):
    fig = plt.figure(figsize=(8,6))
    plt.autoscale()
    linestyles = ["solid", "dashdot", "dotted", "dashed"]
    # colours = ["Navy", "maroon", "orchid", "forestgreen", "lightcoral", "orange",
    #            "darkcyan", "darkviolet"]
    mol_label = ''
    legend_el=[]
    if single:
        for i, mol in enumerate(species):
            print(f'c{len(chem_dfs[0][mol])} t{len(thermal_dfs[0]["Number density"])}')
            offset = abs(len(chem_dfs[0]["Index"]) - len(thermal_dfs[0]["Cell index"]))
            print(offset)
            if mol_name_legend : mol_label = aux.latexify(mol)
            mol_frac = np.divide(
                    chem_dfs[0][mol], thermal_dfs[0]["Number density"][offset:]
                )
            plt.plot(mol_frac, thermal_dfs[0]["Pressure"][offset:], label= mol_label )
    else:
        for i, chem_df in enumerate(chem_dfs):
            offset = abs(len(chem_dfs[i]["Index"]) - len(thermal_dfs[i]["Cell index"]))
            col = CB_color_cycle[i]
            legend_el.append(Line2D([0],[0], color = col, label = labels[i]))
            for j, mol in enumerate(species):
                chem_off = len(chem_df[mol]) - offset
                mol_frac = np.divide(chem_df[mol], chem_df["Number density"])    
                if mol_name_legend : mol_label = aux.latexify(mol)
                
                if(j==0):
                    plt.plot(
                        mol_frac[offset:],
                        thermal_dfs[i]["Pressure"],
                        label= labels[i],
                        c = col,
                        linestyle = linestyles[j]
                    )
                else:
                    plt.plot(
                        mol_frac[offset:],
                        thermal_dfs[i]["Pressure"],
                        c = col,
                        linestyle = linestyles[j]
                        #label = labels[i] + mol
                    )
            if i == 0:
                plt.legend()

    if(len(error_bars) > 0): 
        for i, label in enumerate(measurement_labels):
            plt.errorbar(measurements_x[i], measurements_y[i], error_bars[i][1], error_bars[i][0], label=label, s = 20)       
    else:  
        for i, label in enumerate(measurement_labels):
            plt.scatter(measurements_x[i], measurements_y[i], label=label, marker='v')
    
    for i, mol in enumerate(species):
        legend_el.append(Line2D([0],[0], color = 'black', linestyle=linestyles[i], label = mol))
    
    plt.legend(handles=legend_el)
    plt.grid(alpha = 0)
    plt.xlabel("Mixing Ratio", fontsize = 18)
    plt.ylabel("Pressure [bar]", fontsize = 18)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1e-10,1)
    plt.gca().invert_yaxis()
    plt.ylim(1e-3, 1e-10)
    plt.title(title, fontsize = 20)
    plt.savefig("Plots/chemdiag_mix_" + ext + ".pdf")
    
    return plt

def plot_chem_with_measurements_nden(
    thermal_dfs,
    chem_dfs,
    ext,
    labels,
    title,
    species,
    measurements_x,
    measurements_y,
    measurement_labels,
    single=False,
    error_bars = []
):
    fig = plt.figure()
    plt.autoscale()
    linestyles = ["solid", "dashdot", "dotted", "dashed"]
    # colours = ["Navy", "maroon", "orchid", "forestgreen", "lightcoral", "orange",
    #            "darkcyan", "darkviolet"]

    if single:
        for i, mol in enumerate(species):
            offset = abs(len(chem_dfs[0]["Index"]) - len(thermal_dfs[0]["Cell index"]))

            plt.plot(chem_dfs[0][mol][offset:], thermal_dfs[0]["Altitude"], label= labels[0] + mol)
    else:
        for i, chem_df in enumerate(chem_dfs):
            offset = abs(len(chem_dfs[i]["Index"]) - len(thermal_dfs[i]["Cell index"]))
            for j, mol in enumerate(species):

                plt.plot(
                    chem_df[mol][offset:],
                    thermal_dfs[i]["Altitude"],
                    label= labels[i] + mol,
                    
                )
            if i == 0:
                plt.legend()

    if(len(error_bars) > 0): 
        for i, label in enumerate(measurement_labels):
            plt.errorbar(measurements_x[i], measurements_y[i], error_bars[i][1], error_bars[i][0], label=label, s = 20)       
    else:  
        for i, label in enumerate(measurement_labels):
            plt.scatter(measurements_x[i], measurements_y[i], label=label, marker='v')

    plt.legend()
    plt.xlabel("Number Density [$cm^{-3}$]")
    plt.ylabel("Altitude [km]")
    plt.xscale("log")
    plt.ylim(0,700)
    plt.xlim(1)
    plt.title(title)
    plt.savefig("Plots/chemdiag_nden_" + ext + ".pdf")
    
    return fig
def plot_alt_vs_p(thermals, thermal_labels, meas, meas_labels, ext):
    plt.figure()
    plt.autoscale()
    
    for i, df in enumerate(thermals):
        plt.plot(df["Altitude"], df["Pressure"], label = thermal_labels[i])
    
    for i, measurement in enumerate(meas):
        plt.plot(measurement[0], measurement[1], label = meas_labels[i])
        
    plt.xlabel("Altitude [km]")
    plt.ylabel("Pressure [bar]")
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.legend()
    plt.savefig("Plots/Press_alt_" + ext + ".pdf")
    
def plot_chem(thermal_dfs, chem_dfs, ext, title, species, single=False):
    plt.figure()
    plt.autoscale()
    
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    # colours = ["Navy", "maroon", "orchid", "forestgreen", "lightcoral", "orange",
    #            "darkcyan", "darkviolet"]

    if single:
        offset = abs(len(chem_dfs[0]["Index"]) - len(thermal_dfs[0]["Cell index"]))
        for i, mol in enumerate(species):
            plt.plot(
                chem_dfs[0][mol][offset:],
                thermal_dfs[0]["Pressure"],
                label=mol,
            )
        plt.legend()
    else:
        for i, chem_df in enumerate(chem_dfs):
            offset = abs(len(chem_dfs[i]["Index"]) - len(thermal_dfs[i]["Cell index"]))
            for j, mol in enumerate(species):
                plt.plot(
                    chem_df[mol][offset:],
                    thermal_dfs[i]["Pressure"],
                    label=mol,
                    linestyle=linestyles[i],
                )
            if i == 0:
                plt.legend()

    plt.xlabel("Number Density [$cm^{-3}$]")
    plt.ylabel("Pressure [bar]")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1e-23)
    plt.gca().invert_yaxis()
    plt.ylim(1e-3, 1e-10)
    plt.title(title)
    plt.savefig("Plots/chemdiag_" + ext + ".pdf")


def plot_chem_compare_single_spec(thermal_dfs, chem_dfs, ext, labels, specie):
    """
    Plots several models, but the same species. Make sure the dataframes are in the correct order
    and in the same as labels
    """
    plt.figure()
    plt.autoscale()

    for i, chem_df in enumerate(chem_dfs):
        offset = abs(len(chem_df["Index"]) - len(thermal_dfs[i]["Cell index"]))
        chem_off = len(chem_df[specie]) - offset
        mol_frac = np.divide(chem_df[specie], chem_df["Number density"])
        plt.plot(
            mol_frac[offset:],
            thermal_dfs[i]["Pressure"],
            label=labels[i],
            color=colours[i],
        )

    plt.legend()
    plt.xlabel("Mixing Ratio")
    plt.ylabel("Pressure [bar]")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1e-10)
    plt.gca().invert_yaxis()
    plt.ylim(1e-3, 1e-10)
    plt.title(" ")
    plt.savefig("Plots/Chemcomp_" + ext + ".pdf")
    return plt

if __name__ == "__main__":

    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["axes.labelsize"] = 13
    matplotlib.rcParams["xtick.labelsize"] = 13
    matplotlib.rcParams["ytick.labelsize"] = 13
    matplotlib.rcParams["lines.linewidth"] = 2
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["axes.linewidth"] = 2.0
    matplotlib.rcParams["xtick.major.width"] = 2.0
    matplotlib.rcParams["xtick.minor.width"] = 2.0
    matplotlib.rcParams["ytick.major.width"] = 2.0
    matplotlib.rcParams["ytick.minor.width"] = 2.0
    matplotlib.rcParams["ytick.major.pad"] = 6
    matplotlib.rcParams["ytick.minor.pad"] = 6
    matplotlib.rcParams["xtick.major.pad"] = 6
    matplotlib.rcParams["xtick.minor.pad"] = 6

    parser = ap.ArgumentParser(description="Plot PT-profile for an atmosphere.")
    parser.add_argument(
        "--ext_in", type=str, nargs="+", help="File extensions of input files"
    )
    parser.add_argument("--ext_out", type=str, help="File extension to name the output")
    parser.add_argument("--t", type=str, help="Title of the plot")
    args = parser.parse_args()
    species = ["CH4", "C2H6", "C2H2", "C2H4", "H2S", "SO2"]
    extract_and_plot_pt(args)
    extract_and_plot_chem(args, species)
