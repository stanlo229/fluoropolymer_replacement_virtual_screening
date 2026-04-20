"""
Stefanis–Panayiotou group-contribution tables for Hansen Solubility Parameters.

References
----------
Primary method:
    Stefanis, E.; Panayiotou, C. Int. J. Thermophys. 2008, 29, 568–585.
Updated parameters used here (Tables A.1 / A.2):
    Stefanis, E.; Panayiotou, C. Int. J. Pharm. 2012, 426, 29–43, Appendix A.

HSP formulas (Eqs. A.2–A.4, 2012 update)
------------------------------------------
    delta_d  = (sum_NiCi_d  + W * sum_MjDj_d  + 959.11) ** 0.4126   [MPa^0.5]
    delta_p  =  sum_NiCi_p  + W * sum_MjDj_p  + 7.6134               [MPa^0.5]
    delta_hb =  sum_NiCi_hb + W * sum_MjDj_hb + 7.7003               [MPa^0.5]

    W = 1 if any second-order group is present, else 0.

Low-value fallback (Eqs. A.5–A.6):
    if delta_p  < 3 MPa^0.5: recompute with constant 2.6560
    if delta_hb < 3 MPa^0.5: recompute with constant 1.3720
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard formula constants (Eqs. A.2–A.4)
CONST_D       = 959.11
POWER_D       = 0.4126
CONST_P       = 7.6134
CONST_HB      = 7.7003

# Low-value fallback constants (Eqs. A.5–A.6)
CONST_P_LOW   = 2.6560
CONST_HB_LOW  = 1.3720
THRESHOLD_LOW = 3.0   # MPa^0.5

# ---------------------------------------------------------------------------
# First-order groups  (Table A.1, Stefanis & Panayiotou 2012)
#
# Each entry: (name, smarts, Ci_d, Ci_p, Ci_hb)
# None = contribution unavailable (***) for that parameter.
#
# Groups are listed in priority order for SMARTS matching.
# Complex / functional groups must appear before generic aliphatic groups
# so that each heavy atom is assigned to exactly one group.
# ---------------------------------------------------------------------------

FIRST_ORDER_GROUPS = [
    # ---- Carboxylic acid ------------------------------------------------
    ("COOH",          "[CX3](=O)[OX2H1]",                     -38.16,   0.7153,  3.8422),

    # ---- Esters (specific α-carbon variants first) ----------------------
    ("CH3COO",        "[CH3X4][CX3](=O)[OX2H0]",              -53.86,  -0.6075,  1.7051),
    ("CH2COO",        "[CH2X4][CX3](=O)[OX2H0]",               89.11,   3.4942,  1.3893),
    ("HCOO",          "[H][CX3](=O)[OX2H0]",                    None,   1.7056,  2.3049),
    ("COO",           "[CX3](=O)[OX2H0][#6]",                  27.57,   3.3401,  1.1999),

    # ---- Amides ---------------------------------------------------------
    ("CONH2",         "[CX3](=O)[NX3H2]",                       -1.22,   5.9361,  5.3646),
    ("CONH",          "[CX3](=O)[NX3H1]",                        None,   4.1000,  3.5000),  # not in table; use NH correction
    ("CON(CH3)2",     "[CX3](=O)[NX3H0]([CH3X4])[CH3X4]",      95.97,   5.5309,  3.2455),
    ("CON",           "[CX3](=O)[NX3H0]",                        None,   3.8000,  2.0000),  # generic tertiary amide fallback

    # ---- Ketones --------------------------------------------------------
    ("CH3CO",         "[CH3X4][CX3](=O)[#6]",                  -29.41,   2.1567, -1.1683),
    ("CH2CO",         "[CH2X4][CX3](=O)[#6]",                  114.74,   3.6103, -0.3929),
    ("Ccyclic_CO",    "[CX3;R](=O)",                              None,    None,    None),   # handled via 2nd-order
    (">C=O_other",    "[CX3](=O)",                             -127.16,   0.7691,  1.7033),

    # ---- Aldehyde -------------------------------------------------------
    ("CHO_ald",       "[CX3H1]=O",                              -31.35,   3.3159,  0.2062),

    # ---- Hydroxyl -------------------------------------------------------
    # ACOH / OH_aliph: group owns only the O atom; neighbouring C is context.
    ("ACOH",          "[OX2H1;$([OX2H1][c])]",                  58.52,   1.0520,  6.9757),
    ("OH_aliph",      "[OX2H1;$([OX2H1][CX4])]",               -29.97,   1.0587,  7.3609),

    # ---- Ethers ---------------------------------------------------------
    ("C2H5O2",        "[CH2X4][OX2H0][CH2X4][OX2H1]",           15.51,   3.3880,  8.5893),
    ("CH2O_cyclic",   "[CH2X4;R][OX2H0;R]",                     49.32,   0.1227,  0.1763),
    ("CH3O",          "[CH3X4][OX2H0]",                         -68.07,   0.0089,  0.2676),
    ("CH2O_ether",    "[CH2X4][OX2H0]",                          13.40,   0.8132, -0.1196),
    ("CHO_ether",     "[CHX4][OX2H0]",                          111.46,   1.6001,  0.4873),
    ("O_other",       "[OX2H0;!$([O]c);!$([O]C=O)]",            18.09,   3.5248,  0.0883),

    # ---- Nitrile --------------------------------------------------------
    ("CH2CN",         "[CH2X4][CX2]#[NX1]",                    -29.09,   6.3586, -0.7297),
    ("CN_other",      "[CX2]#[NX1]",                             49.36,   6.3705, -0.5239),

    # ---- Isocyanate -----------------------------------------------------
    ("OCN",           "[OX1]=[CX2]=[NX2]",                      15.22,   1.4695,  4.1129),

    # ---- Nitro ----------------------------------------------------------
    ("CH2NO2",        "[CH2X4][NX3+](=O)[O-]",                   None,   6.6451, -1.0669),
    ("CHNO2",         "[CHX4][NX3+](=O)[O-]",                    None,   7.7753, -2.1087),
    # ACNO2: owns N+2O; aromatic C is context only.
    ("ACNO2",         "[NX3+;$([NX3+](=O)[O-][c])](=O)[O-]",   219.22,   4.4640, -0.7302),

    # ---- Sulfur ---------------------------------------------------------
    ("SO2",           "[SX4](=O)(=O)",                          182.83,  11.0254, -0.3602),
    (">C=S",          "[CX3]=S",                                  -0.46,   0.5216,  3.0519),
    ("CH2SH",         "[CH2X4][SX2H1]",                         214.84,  -0.9940,  4.5321),
    ("SH_other",      "[SX2H1]",                                190.87,   1.8229,  4.9279),
    ("CH3S",          "[CH3X4][SX2H0]",                           None,   0.2451, -1.2669),
    ("CH2S",          "[CH2X4][SX2H0]",                         168.57,   0.5730, -0.0838),
    ("S_other",       "[SX2H0]",                                201.91,   8.5982, -0.4013),

    # ---- Amines ---------------------------------------------------------
    # ACNH2: owns the N only; aromatic C is counted separately as ACH/AC.
    ("ACNH2",         "[NX3H2;$([NX3H2][c])]",                 253.66,   1.6493,  4.4945),
    ("CH2NH2",        "[CH2X4][NX3H2]",                         -49.96,  -0.3449,  2.7280),
    ("CHNH2",         "[CHX4][NX3H2]",                           18.53,  -1.4337,  0.5647),
    ("CH3NH",         "[CH3X4][NX3H1]",                           None,   0.5060,  5.7321),
    ("CH2NH",         "[CH2X4][NX3H1]",                          96.18,   0.2616,  1.4053),
    ("NH_other",      "[NX3H1;!$(NC=O);!$(Nc)]",                  None,  -0.0746,  2.0646),
    ("CH3N",          "[CH3X4][NX3H0]",                         170.59,   1.0575,  1.8500),
    ("CH2N",          "[CH2X4][NX3H0]",                         152.54,   2.6766,  1.5557),
    ("N_other",       "[NX3H0;!$(NC=O);!$(Nc)]",                267.06,   2.2212,  1.3655),

    # ---- Imine / pyridine -----------------------------------------------
    (">C=N_",         "[CX3]=[NX2]",                            -10.55,  -0.1692, -5.3820),
    ("-CH=N-",        "[CHX3]=[NX2]",                           186.40,   2.7015,  0.5507),

    # ---- Halogens -------------------------------------------------------
    ("CF3",           "[CX4H0]([F])([F])[F]",                   -13.79,  -2.1381, -1.2997),
    ("CF2",           "[CX4H0]([F])[F]",                       -103.83,    None,    None),
    ("ACF",           "[FX1][c]",                                27.74,   0.1293, -0.6613),
    ("F_other",       "[FX1]",                                  -80.11,    None,    None),
    ("CCl3",          "[CX4H0]([ClX1])([ClX1])[ClX1]",           None,   1.1060, -2.5679),
    ("CHCl2",         "[CHX4]([ClX1])[ClX1]",                  197.67,   1.6255, -3.0669),
    ("CCl2",          "[CX4H0]([ClX1])[ClX1]",                  72.60,   0.1035, -1.3220),
    ("CCl",           "[CX4H0][ClX1]",                         385.39,   1.8196,  0.1473),
    ("CHCl",          "[CHX4][ClX1]",                            73.01,   2.6796, -1.3563),
    ("CH2Cl",         "[CH2X4][ClX1]",                           47.17,   0.5013, -0.4498),
    ("ACCl",          "[ClX1][c]",                              141.54,  -0.0941, -0.7512),
    ("Cl_other",      "[ClX1]",                                  76.35,   1.7491, -0.2917),
    ("ACBr",          "[BrX1][c]",                                None,    None,    None),   # via 2nd-order ACBr
    ("Br",            "[BrX1]",                                 109.79,   0.5207, -0.9087),
    ("I",             "[IX1]",                                  197.67,   0.1060,  0.3321),

    # ---- Aromatic carbons (match after aromatic substituents above) ------
    ("ACCH3",         "[cH0]-[CH3X4]",                           27.67,  -0.6212, -1.1409),
    ("ACCH2",         "[cH0]-[CH2X4]",                           89.07,   0.8019, -0.2298),
    ("ACH",           "[cH1]",                                   29.87,  -0.5771, -0.3554),
    ("AC",            "[cH0]",                                   98.84,   0.7661, -0.1553),

    # ---- Olefinic carbons -----------------------------------------------
    ("CH2_CH_",       "[CH2X3]=[CHX2]",                        -126.15,  -2.0170, -1.1783),
    ("-CH_CH-",       "[CHX3]=[CHX3]",                           28.65,  -0.5037, -0.1253),
    ("CH2_C_",        "[CH2X3]=[CX3H0]",                        -31.62,  -0.9052, -0.7191),
    ("-CH_C_",        "[CHX3]=[CX3H0]",                          62.48,  -1.1018, -1.7171),
    (">C_C_",         "[CX3H0]=[CX3H0]",                         50.10,   0.9957, -1.9773),
    ("CH2_C_CH-",     "[CH2X3]=[CX2]=[CHX2]",                  -161.71,   None,  -0.7545),

    # ---- Alkyne ---------------------------------------------------------
    ("CH_trp_C",      "[CX2H1]#[CX2]",                           45.86,  -1.5147,  1.2582),
    ("C_trp_C",       "[CX2H0]#[CX2H0]",                          9.56,  -0.9552, -1.0176),

    # ---- Aliphatic carbons (lowest priority) ----------------------------
    ("-CH3",          "[CH3X4]",                               -123.01,  -1.6444, -0.7458),
    ("-CH2-",         "[CH2X4]",                                  1.82,  -0.3141, -0.3877),
    ("-CH<",          "[CHX4]",                                  82.94,   0.6051, -0.2064),
    (">C<",           "[CX4H0]",                                182.13,   2.0249, -0.0113),
]

# ---------------------------------------------------------------------------
# Second-order groups  (Table A.2, Stefanis & Panayiotou 2012)
#
# Each entry: (name, Dj_d, Dj_p, Dj_hb, identify_fn)
# identify_fn(mol, used_fo_groups) -> int (number of occurrences)
#
# Identification is done structurally after first-order group counting.
# None = contribution unavailable (***).
# ---------------------------------------------------------------------------

# Second-order group data: (name, Dj_d, Dj_p, Dj_hb)
# Identification logic is implemented in hsp_calculator.py
SECOND_ORDER_GROUPS = [
    ("(CH3)2-CH-",          7.63,    0.0365,   0.3019),
    ("(CH3)3-C-",          -0.03,    1.1593,  -0.1924),
    ("ring_5C",           -81.93,   -2.3673,   0.2586),
    ("ring_6C",           -26.15,   -3.6661,    None),
    ("ring_3C",            15.35,    1.7962,  -0.7224),
    ("-C=C-C=C-",          -5.69,   -3.3100,  -1.2207),
    ("CH3-C=",             -4.45,    0.3461,   0.4418),
    ("-CH2-C=",           -29.67,   -2.3189,  -0.5613),
    (">C{H/C}-C=",         -5.38,     None,   -1.0241),
    ("string_in_cyclic",  -54.05,     None,     None),
    ("CH3(CO)CH2-",         3.57,   -0.4108,  -0.3628),
    ("Ccyclic=O",         -46.57,    0.1972,  -0.4496),
    ("ACCOOH",            -37.57,   -0.6284,  -0.8552),
    (">C{H/C}-COOH",        None,   -0.2450,   1.2554),
    ("CH3(CO)OC{H/C}<",   -40.56,   -0.0652,   0.3864),
    ("(CO)C{H2}COO",        None,   -2.3624,   0.8545),
    ("(CO)O(CO)",         -92.46,   -0.9818,   1.5759),
    ("ACHO",               46.84,   -1.8120,  -0.9192),
    (">CHOH",              16.54,    0.2366,  -0.2453),
    (">C<OH",              -5.97,   -0.0069,   1.3813),
    ("-C(OH)C(OH)-",        None,    0.6669,   0.2493),
    ("-C(OH)C(N)",         -7.03,    0.8750,  -0.7322),
    ("Ccyclic-OH",         -6.40,   -3.6065,   0.5836),
    ("C-O-C=C",            25.23,    0.5480,   1.1279),
    ("AC-O-C",             35.82,    0.7781,   0.6689),
    ("ACCOO",             -38.39,    0.3670,  -0.2340),
    ("AC-O-AC",          -136.10,   -3.4995,   1.8763),
    (">N{H/C}(cyclic)",    53.29,   -1.6876,  -0.0132),
    ("-S-(cyclic)",        91.57,    0.2513,   0.2663),
    ("ACBr",               33.35,   -0.4478,   0.3149),
    ("(C=C)-Br",          -85.85,    0.0686,  -1.1154),
    ("AC(ACHm)2AC(ACHn)2", -33.14,  -1.4784,   0.7468),
    ("Ocyclic-Ccyclic=O",  13.89,    2.7261,   0.2185),
    ("NcyclicH-Ccyclic=O", 93.54,    2.0813,   1.2226),
    ("-O-CHm-O-CHn-",      31.52,    0.3293,   0.2527),
    ("C(=O)-C-C(=O)",     -61.38,   -0.4126,   1.2240),
]

# ---------------------------------------------------------------------------
# Low δp / low δhb first-order corrections (Tables A.5 & A.6, 2012)
#
# Same SMARTS as FIRST_ORDER_GROUPS; different Ci values when
# predicted delta_p < 3 or delta_hb < 3 MPa^0.5.
# Keyed by group name → (Ci_p_low, Ci_hb_low)
# None = not available for that parameter.
# ---------------------------------------------------------------------------

LOW_VALUE_CORRECTIONS = {
    "-CH3":           (-0.7107,  0.2990),
    "-CH2-":          (-0.1361, -0.1161),
    "-CH<":           ( 0.6477,  0.1386),
    ">C<":            (   None, -0.1212),
    "CH2_CH_":        (-0.2511,  1.3552),
    "-CH_CH-":        (-0.1503,  0.4819),
    "CH2_C_":         ( 0.6956,  0.1115),
    "-CH_C_":         ( 1.2761, -0.0307),
    ">C_C_":          (   None, -0.1212),
    "CH2_C_CH-":      (-0.2453,    None),
    "CH_trp_C":       (-0.7049,  0.4385),
    "C_trp_C":        (   None, -0.3511),
    "ACH":            (-0.1930,  0.1353),
    "AC":             ( 0.1745, -0.1740),
    "ACCH3":          (-0.4493, -0.2873),
    "ACCH2":          (-0.2857, -0.8808),
    "COOH":           ( 2.9098,    None),
    "CH3COO":         ( 1.7711,    None),
    "CH2COO":         ( 2.2096,    None),
    "COO":            ( 1.4783,  0.3720),
    "OH_aliph":       (   None,    None),
    "CH3O":           (-0.3600,    None),
    "CH2O_ether":     (   None,    None),
    "CHO_ether":      (   None, -0.4067),
    "CH2O_cyclic":    (-0.2919,    None),
    "CH2NH2":         (   None,    None),
    "CH2NH":          ( 0.8875,    None),
    "CH3N":           (   None, -0.1700),
    "CH2N":           ( 0.6477, -1.0369),  # closest match; Table A.5 shows 0.6477 not 0.7055
    "CH2S":           (   None,  0.1461),
    "CH2Cl":          (   None,  0.4895),
    "CHCl":           (   None,  0.1300),
    "CHCl2":          (   None,  0.5254),
    "ACCl":           (-0.0927,  0.4424),
    "ACF":            (   None, -0.3718),
    "Cl_other":       (   None,  1.1251),
    "CF3":            (   None, -0.0887),
    "O_other":        (-0.5555,    None),
    "S_other":        ( 0.0445,    None),
    ">C=O_other":     (   None, -0.0553),
}

# Low-δ second-order corrections (Table A.6, 2012)
# Keyed by second-order group name → (Dj_p_low, Dj_hb_low)
LOW_VALUE_2ND_ORDER = {
    "(CH3)2-CH-":           ( 0.2246,  0.0000),
    "ring_5C":              (-0.9657,  0.1944),
    "ring_6C":              (-0.9615,  0.0000),
    "-C=C-C=C-":            ( 0.6463,    None),
    "CH3-C=":               (-0.0063, -0.0614),
    "-CH2-C=":              ( 0.0192,  0.0660),
    ">C{H/C}-C=":           (-0.4460,  0.3422),
    "string_in_cyclic":     (   None, -0.2809),
    "ACCOO":                ( 0.4912,  0.0000),
    "AC(ACHm)2AC(ACHn)2":   ( 0.0130,  0.0864),
    "-O-CHm-O-CHn-":        ( 0.0000,    None),
}
