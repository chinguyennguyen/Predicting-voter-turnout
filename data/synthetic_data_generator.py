import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------
# Paths (relative to this file)
# --------------------------------------
here = Path(__file__).resolve().parent          # .../Predicting-voter-turnout/data
output_path = here / "synthetic_data.dta"

# --------------------------------------
# Real Swedish kommun codes (4-digit rows from your list)
# --------------------------------------
kommun_codes = [
    "0114","0115","0117","0120","0123","0125","0126","0127","0128","0136",
    "0138","0139","0140","0160","0162","0163","0180","0181","0182","0183",
    "0184","0186","0187","0188","0191","0192",
    "0305","0319","0330","0331","0360","0380","0381","0382",
    "0428","0461","0480","0481","0482","0483","0484","0486","0488",
    "0509","0512","0513","0560","0561","0562","0563","0580","0581","0582",
    "0583","0584","0586",
    "0604","0617","0642","0643","0662","0665","0680","0682","0683","0684",
    "0685","0686","0687",
    "0760","0761","0763","0764","0765","0767","0780","0781",
    "0821","0834","0840","0860","0861","0862","0880","0881","0882","0883",
    "0884","0885",
    "0980",
    "1060","1080","1081","1082","1083",
    "1214","1230","1231","1233","1256","1257","1260","1261","1262","1263",
    "1264","1265","1266","1267","1270","1272","1273","1275","1276","1277",
    "1278","1280","1281","1282","1283","1284","1285","1286","1287","1290",
    "1291","1292","1293",
    "1315","1380","1381","1382","1383","1384",
    "1401","1402","1407","1415","1419","1421","1427","1430","1435","1438",
    "1439","1440","1441","1442","1443","1444","1445","1446","1447","1452",
    "1460","1461","1462","1463","1465","1466","1470","1471","1472","1473",
    "1480","1481","1482","1484","1485","1486","1487","1488","1489","1490",
    "1491","1492","1493","1494","1495","1496","1497","1498","1499",
    "1715","1730","1737","1760","1761","1762","1763","1764","1765","1766",
    "1780","1781","1782","1783","1784","1785",
    "1814","1860","1861","1862","1863","1864","1880","1881","1882","1883",
    "1884","1885",
    "1904","1907","1960","1961","1962","1980","1981","1982","1983","1984",
    "2021","2023","2026","2029","2031","2034","2039","2061","2062","2080",
    "2081","2082","2083","2084","2085",
    "2101","2104","2121","2132","2161","2180","2181","2182","2183","2184",
    "2260","2262","2280","2281","2282","2283","2284",
    "2303","2305","2309","2313","2321","2326","2361","2380",
    "2401","2403","2404","2409","2417","2418","2421","2422","2425","2460",
    "2462","2463","2480","2481","2482",
    "2505","2506","2510","2513","2514","2518","2521","2523","2560","2580",
    "2581","2582","2583","2584"
]

# simple placeholder names; you can merge real names in Stata later if you need to
kommun_namn = [f"Kommun_{code}" for code in kommun_codes]

kommuner = pd.DataFrame({
    "Kommun": kommun_codes,
    "Kommun_namn": kommun_namn
})
# Lan = first two digits of Kommun
kommuner["Lan"] = kommuner["Kommun"].str[:2]

# --------------------------------------
# Number of obs & seed
# --------------------------------------
n = 40000 
np.random.seed(12345)

df = pd.DataFrame(index=np.arange(n))

# --------------------------------------
# ID & fgang18
# --------------------------------------
df["P1401_LopNr_PersonNr"] = df.index + 1
df["fgang18"] = np.random.binomial(1, 0.0549, size=n)

# --------------------------------------
# Kommun, Kommun_namn, Lan
# --------------------------------------
chosen = kommuner.sample(n, replace=True).reset_index(drop=True)
df["Kommun"] = chosen["Kommun"]
df["Kommun_namn"] = chosen["Kommun_namn"]
df["Lan"] = chosen["Lan"]  # first two digits of Kommun

# --------------------------------------
# district18 = Kommun + 4 random digits
# --------------------------------------
digits4 = np.random.randint(0, 10000, size=n).astype(str)
digits4 = np.char.zfill(digits4, 4)
df["district18"] = df["Kommun"] + digits4

# --------------------------------------
# y_mun ("VV", "VN", "NV", "NN") – TARGET_COLUMN
# --------------------------------------
y = np.array([""] * n, dtype=object)
u1 = np.random.rand(n)
y[u1 <= 0.09] = "NN"

mask = y == ""
u2 = np.random.rand(n)
y[mask & (u2 <= 0.043 / (1 - 0.09))] = "NV"

mask = y == ""
u3 = np.random.rand(n)
y[mask & (u3 <= 0.076 / (1 - 0.09 - 0.043))] = "VN"

y[y == ""] = "VV"
df["y_mun"] = y   # <-- TARGET_COLUMN

# --------------------------------------
# helper for truncated normal integers
# --------------------------------------
def truncnorm_int(mean, sd, low, high, size):
    x = np.round(np.random.normal(mean, sd, size))
    return np.clip(x, low, high).astype(int)

# baseline byear (will adjust later by marital_status)
df["byear"] = truncnorm_int(1969, 19, 1912, 2000, n)

# female
df["female"] = np.random.binomial(1, 0.504, size=n)

# schooling_years
df["schooling_years"] = truncnorm_int(12.21, 2.589, 7, 20, n)

# --------------------------------------
# Country: fodelslandnamn_Grp (as in your Stata logic)
# --------------------------------------
u_country = np.random.rand(n)
country = np.array([""] * n, dtype=object)

country[u_country <= 0.70] = "SVERIGE"
country[(u_country > 0.70) & (u_country <= 0.75)] = "FINLAND"
country[(u_country > 0.75) & (u_country <= 0.80)] = "IRAK"
country[(u_country > 0.80) & (u_country <= 0.84)] = "POLEN"
country[(u_country > 0.84) & (u_country <= 0.88)] = "SYRIEN"
country[(u_country > 0.88) & (u_country <= 0.91)] = "IRAN"
country[(u_country > 0.91) & (u_country <= 0.93)] = "JUGOSLAVIEN"
country[(u_country > 0.93) & (u_country <= 0.95)] = "BOSNIEN OCH HERCEGOVINA"
country[(u_country > 0.95) & (u_country <= 0.96)] = "SOMALIA"
country[(u_country > 0.96) & (u_country <= 0.97)] = "TURKIET"
country[(u_country > 0.97) & (u_country <= 0.98)] = "TYSKLAND"
country[(u_country > 0.98) & (u_country <= 0.985)] = "THAILAND"
country[(u_country > 0.985) & (u_country <= 0.99)] = "NORGE"
country[(u_country > 0.99) & (u_country <= 0.993)] = "DANMARK"
country[(u_country > 0.993) & (u_country <= 0.995)] = "CHILE"
country[(u_country > 0.995) & (u_country <= 0.997)] = "AFGHANISTAN"
country[u_country > 0.997] = "LIBANON"

df["fodelslandnamn_Grp"] = country

# foreigner: dummy 1 if not born in Sweden
df["foreigner"] = (df["fodelslandnamn_Grp"] != "SVERIGE").astype(int)

# birth_continent (as in your Stata mapping)
birth_cont = np.array([""] * n, dtype=object)
c = df["fodelslandnamn_Grp"].values

birth_cont[c == "SVERIGE"] = "Sverige"
birth_cont[np.isin(c, ["FINLAND", "NORGE", "DANMARK"])] = "Norden utom Sverige"
birth_cont[np.isin(c, ["POLEN", "TYSKLAND"])] = "EU27 utom Norden"
birth_cont[np.isin(c, ["JUGOSLAVIEN", "BOSNIEN OCH HERCEGOVINA"])] = "Europa utom EU27 utom Norden"
birth_cont[np.isin(c, ["IRAK", "SYRIEN", "IRAN", "THAILAND",
                       "AFGHANISTAN", "LIBANON", "TURKIET"])] = "Asien"
birth_cont[c == "SOMALIA"] = "Afrika"
birth_cont[(c == "CHILE") | (birth_cont == "")] = "others"

df["birth_continent"] = birth_cont

# --------------------------------------
# Children counts
# --------------------------------------
df["barn0_6"] = truncnorm_int(0.2009, 0.539, 0, 6, n)
df["barn7_17"] = truncnorm_int(0.3247, 0.7168, 0, 11, n)
df["barn_above18"] = truncnorm_int(0.25624, 0.5838, 0, 8, n)
df["total_barn"] = df["barn0_6"] + df["barn7_17"] + df["barn_above18"]

# --------------------------------------
# marital_status
# --------------------------------------
u = np.random.rand(n)
mar = np.empty(n, dtype=object)
mar[u <= 0.42] = "married"
mar[(u > 0.42) & (u <= 0.83)] = "single"
mar[(u > 0.83) & (u <= 0.95)] = "separated"
mar[(u > 0.95) & (u <= 0.99)] = "widowed"
mar[u > 0.99] = "others"
df["marital_status"] = mar

# --------------------------------------
# employment_status
# --------------------------------------
u = np.random.rand(n)
emp = np.empty(n, dtype=object)
emp[u <= 0.64] = "employed"
emp[(u > 0.64) & (u <= 0.97)] = "nilf"
emp[u > 0.97] = "unemployed"
df["employment_status"] = emp

# --------------------------------------
# Income variables
# --------------------------------------
# total_income
df["total_income"] = np.random.normal(3239.8, 2503, size=n)
df.loc[df["employment_status"] == "employed", "total_income"] *= 1.5
df.loc[df["employment_status"] == "nilf", "total_income"] *= 0.8
df.loc[df["employment_status"] == "unemployed", "total_income"] *= 0.5
df["total_income"] = df["total_income"].clip(0, 1014696)

# capital_income
df["capital_income"] = np.random.normal(262, 2810, size=n)

# share_labor_income (0–1, truncated)
mean_labor, sd_labor = 0.6154, 0.438
share_labor = np.random.normal(mean_labor, sd_labor, size=n)
share_labor = np.clip(share_labor, 0, 1)
df["share_labor_income"] = share_labor

# --------------------------------------
# hh_received_sa & share_sa
# --------------------------------------
df["hh_received_sa"] = np.random.binomial(1, 0.15, size=n)
share_sa = np.zeros(n)
mask_sa = df["hh_received_sa"] == 1
share_sa[mask_sa] = np.random.normal(0.4, 0.2, size=mask_sa.sum())
share_sa[~mask_sa] = np.random.normal(0.0002, 0.001, size=(~mask_sa).sum())
df["share_sa"] = np.clip(share_sa, 0, 1)

# --------------------------------------
# Adjust byear by marital_status (target ages)
# --------------------------------------
mask_wid = df["marital_status"] == "widowed"
df.loc[mask_wid, "byear"] = truncnorm_int(2018 - 73, 10, 1912, 2000, mask_wid.sum())

mask_single = df["marital_status"] == "single"
df.loc[mask_single, "byear"] = truncnorm_int(2018 - 36, 8, 1912, 2000, mask_single.sum())

mask_other = df["marital_status"].isin(["married", "separated", "others"])
df.loc[mask_other, "byear"] = truncnorm_int(2018 - 49, 15, 1912, 2000, mask_other.sum())

df["age_2018"] = 2018 - df["byear"]

# --------------------------------------
# sector (based on employment_status)
# --------------------------------------
sector = np.array(["no_job"] * n, dtype=object)
mask_emp = df["employment_status"] == "employed"
u = np.random.rand(mask_emp.sum())
s = np.empty(mask_emp.sum(), dtype=object)
s[u <= 0.45] = "private_company"
s[(u > 0.45) & (u <= 0.70)] = "government_admin"
s[(u > 0.70) & (u <= 0.85)] = "public_company"
s[(u > 0.85) & (u <= 0.95)] = "self_employed"
s[u > 0.95] = "others"
sector[mask_emp] = s
df["sector"] = sector

# --------------------------------------
# Distriktskod = Kommun + 2 random digits
# --------------------------------------
digits2 = np.random.randint(0, 100, size=n).astype(str)
digits2 = np.char.zfill(digits2, 2)
df["Distriktskod"] = df["Kommun"] + digits2

# --------------------------------------
# Coerce non-numeric columns to plain object dtype for Stata
# --------------------------------------
for col in df.columns:
    dt = df[col].dtype
    is_num = False
    try:
        is_num = np.issubdtype(dt, np.number)
    except TypeError:
        is_num = False
    if not is_num:
        df[col] = df[col].astype(object)

# --------------------------------------
# Save to Stata
# --------------------------------------
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_stata(output_path, write_index=False, version=118)

print(f"Saved {output_path} with {len(df)} observations")
