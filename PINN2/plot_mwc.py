import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_year = 1991
end_year = 2025
depth_inter = 350

#Beaufort Gyre,buffer zone,  Pan Arctic
lat_n = 80.5 #90.5 #90 
lat_s = 70.5 #75 #50
lon_e = -130 #-10 #180
lon_w = -170 #-130 #-180

region = "Beaufort_Gyre"

Beaufort_Gyre = {"lat_n":80.5,"lat_s":70.5}

df = pd.read_parquet(f'infer_output/MWC/mwc_siwc_{start_year}_{end_year}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.parquet')

# Extract year and month
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")

# Pivot so each month becomes its own series
pivot_df = df.pivot(index="year", columns="month_name", values="mwc")
# Pivot so each month becomes its own series
pivot_df_siwc = df.pivot(index="year", columns="month_name", values="siwc")
# Ensure month order is correct
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

pivot_df = pivot_df[month_order]
pivot_df_siwc = pivot_df_siwc[month_order]
# Annual mean
annual_mean_mwc = df.groupby("year")["mwc"].mean()
annual_mean_siwc = df.groupby("year")["siwc"].mean()

annual_mean = annual_mean_mwc + annual_mean_siwc

# Plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)

# Plot one line per month
for month in month_order:
    ax.plot(
        pivot_df.index,
        pivot_df[month],
        linewidth=1.5,
        label=month
    )

# Plot annual mean
ax.plot(
    annual_mean_mwc.index,
    annual_mean_mwc.values,
    color="black",
    linewidth=3,
    linestyle="--",
    label="Annual Mean"
)

ax.set_title("Average Meteoric water Content (km³)")
ax.set_xlabel("Year")
ax.set_ylabel("Meteoric water Content (km³)")

# Legend
ax.legend(
    title="Month",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"infer_output/MWC/mean_mwc_{start_year}_{end_year}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.png", dpi=150)
plt.close(fig)

# Plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)

# Plot one line per month
for month in month_order:
    ax.plot(
        pivot_df_siwc.index,
        pivot_df_siwc[month],
        linewidth=1.5,
        label=month
    )

# Plot annual mean
ax.plot(
    annual_mean_siwc.index,
    annual_mean_siwc.values,
    color="black",
    linewidth=3,
    linestyle="--",
    label="Annual Mean"
)

ax.set_title("Average Sea Ice melt water Content (km³)")
ax.set_xlabel("Year")
ax.set_ylabel("Sea Ice melt water Content (km³)")

# Legend
ax.legend(
    title="Month",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"infer_output/MWC/mean_siwc_{start_year}_{end_year}_{lat_n}_{lat_s}_{lon_e}_{lon_w}.png", dpi=150)
plt.close(fig)


# # plot comparison with Proshutinsky 2019
# Beaufort Gyre
if lat_n != 80.5 or lat_s != 70.5 or lon_e != -130 or lon_w != -170:
    raise ValueError(f"Lat/Lon extent is outside Beaufort Gyre! "
                     f"Got: lat_n={lat_n}, lat_s={lat_s}, lon_e={lon_e}, lon_w={lon_w}")
pro = pd.read_csv("FWC_Fig3a_Proshutinsky_2019.csv")
pro.columns = pro.columns.str.strip()


def decimal_year_to_datetime(decimal_years):
    def convert(y):
        year = int(y)
        remainder = y - year
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        return start + (end - start) * remainder
    return pd.Series([convert(y) for y in decimal_years])

pro["Year"] = decimal_year_to_datetime(pro["Year"])

def zscore(s): return (s - s.mean()) / s.std()

# Group Proshutinsky monthly data into annual means
pro["YearInt"] = pro["Year"].dt.year
pro_satellites = pro.groupby("YearInt")["FWC1"].mean()
pro_ctd = pro.groupby("YearInt")["FWC2"].mean()

# Convert year index to datetime for plotting
pro_satellites.index = pd.to_datetime(pro_satellites.index, format='%Y')
pro_ctd.index = pd.to_datetime(pro_ctd.index, format='%Y')

# Filter
annual_mean_filtered = annual_mean[(annual_mean.index >= 2003) & (annual_mean.index <= 2014)]
annual_mean_filtered.index = pd.to_datetime(annual_mean_filtered.index, format='%Y')
annual_mean_mwc_filtered = annual_mean_mwc[(annual_mean_mwc.index >= 2003) & (annual_mean_mwc.index <= 2014)]
annual_mean_mwc_filtered.index = pd.to_datetime(annual_mean_mwc_filtered.index, format='%Y')
annual_mean_siwc_filtered = annual_mean_siwc[(annual_mean_siwc.index >= 2003) & (annual_mean_siwc.index <= 2014)]
annual_mean_siwc_filtered.index = pd.to_datetime(annual_mean_siwc_filtered.index, format='%Y')

# Normalise
annual_mean_norm = zscore(annual_mean_filtered.values)
pro_sats_norm = zscore(pro_satellites.values)
pro_ctd_norm = zscore(pro_ctd.values)
annual_mean_mwc_norm = zscore(annual_mean_mwc_filtered.values)
annual_mean_siwc_norm = zscore(annual_mean_siwc_filtered.values)
# Plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(annual_mean_filtered.index, annual_mean_norm,
        color="black", linewidth=3, linestyle="-", label="Annual Mean FWC")
ax.plot(pro_satellites.index, pro_sats_norm,
        color="red", linewidth=3, linestyle="-", label="Pro Sats")
ax.plot(pro_ctd.index, pro_ctd_norm,
        color="blue", linewidth=3, linestyle="-", label="Pro CTD's")
ax.plot(annual_mean_mwc_filtered.index, annual_mean_mwc_norm,
        color="green", linewidth=2, linestyle="--", label="MWC contribution")
ax.plot(annual_mean_siwc_filtered.index, annual_mean_siwc_norm,
        color="orange", linewidth=2, linestyle="--", label="SIWC contribution")

ax.set_title("Compare Proshutinsky with FWC (Normalised)")
ax.set_xlabel("Year")
ax.set_ylabel("Freshwater/Meteoric Content (normalised)")

# Legend
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("infer_output/MWC/fwc_proshutinsky_compare_norm.png", dpi=150)
plt.close(fig)

# Plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(annual_mean_filtered.index, annual_mean_filtered/1000,
        color="black", linewidth=3, linestyle="-", label="Annual Mean FWC")
ax.plot(pro_satellites.index, pro_satellites,
        color="red", linewidth=3, linestyle="-", label="Pro Sats")
ax.plot(pro_ctd.index, pro_ctd,
        color="blue", linewidth=3, linestyle="-", label="Pro CTD's")
ax.plot(annual_mean_mwc_filtered.index, annual_mean_mwc_filtered/1000,
        color="green", linewidth=2, linestyle="--", label="MWC contribution")
ax.plot(annual_mean_siwc_filtered.index, annual_mean_siwc_filtered/1000,
        color="orange", linewidth=2, linestyle="--", label="SIWC contribution")

ax.set_title("Compare Proshutinsky with FWC")
ax.set_xlabel("Year")
ax.set_ylabel("Freshwater/Meteoric Content")

# Legend
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("infer_output/MWC/fwc_proshutinsky_compare.png", dpi=150)
plt.close(fig)

# MWC anomoly plot
# Compute reference mean (1991–2004)
reference_mean_mwc = annual_mean_mwc[(annual_mean_mwc.index >= 1991) & (annual_mean_mwc.index <= 2004)].mean()

# Compute anomaly
anomaly = annual_mean_mwc - reference_mean_mwc

# Plot
fig, ax = plt.subplots(figsize=(16, 10))

colors = ["red" if v > 0 else "blue" for v in anomaly]
ax.bar(anomaly.index, anomaly.values, color=colors, alpha=0.7)
ax.axhline(0, color="black", linewidth=1)

# Shade reference period
ax.axvspan(1991, 2004, color="grey", alpha=0.15, label="Reference period (1991–2004)")

ax.set_title("Meteoric water Content Anomaly (reference: 1991–2004 mean)")
ax.set_xlabel("Year")
ax.set_ylabel("MWC Anomaly (km^3)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("infer_output/MWC/mwc_anomaly.png", dpi=150)
plt.close(fig)

# SIWC anomoly plot
# Compute reference mean (1991–2004)
reference_mean_siwc = annual_mean_siwc[(annual_mean_siwc.index >= 1991) & (annual_mean_siwc.index <= 2004)].mean()

# Compute anomaly
anomaly = annual_mean_siwc - reference_mean_siwc

# Plot
fig, ax = plt.subplots(figsize=(16, 10))

colors = ["red" if v > 0 else "blue" for v in anomaly]
ax.bar(anomaly.index, anomaly.values, color=colors, alpha=0.7)
ax.axhline(0, color="black", linewidth=1)

# Shade reference period
ax.axvspan(1991, 2004, color="grey", alpha=0.15, label="Reference period (1991–2004)")

ax.set_title("Sea Ice melt water Content Anomaly (reference: 1991–2004 mean)")
ax.set_xlabel("Year")
ax.set_ylabel("SIWC Anomaly (km^3)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("infer_output/MWC/siwc_anomaly.png", dpi=150)
plt.close(fig)