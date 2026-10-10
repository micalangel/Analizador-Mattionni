"""Catálogo de activos con nombre legible: índices, acciones, divisas, materias primas y criptomonedas.

Acciones: las mayores de EE. UU., todo el IBEX 35 y buena parte del mercado continuo español,
Euro Stoxx 50 y grandes europeas, Japón, China, Corea, Taiwán, India, Australia y Latinoamérica,
más algunas empresas pequeñas o recientes muy seguidas (Rivian, SoFi, Rocket Lab, SpaceX…).

La web muestra siempre el nombre ("Amazon", "Santander", "Euro / Dólar") y busca por nombre,
país o ticker. Cualquier otro ticker de Yahoo Finance sigue funcionando.
"""

from __future__ import annotations

import re
import unicodedata

from .indices import INDICES

# ticker: (nombre, tipo, país, sector)
ACTIVOS: dict[str, tuple[str, str, str, str]] = {
    # --- Acciones: EE. UU. ---
    "AAPL": ("Apple", "Acción", "Estados Unidos", "Tecnología"),
    "MSFT": ("Microsoft", "Acción", "Estados Unidos", "Tecnología"),
    "NVDA": ("Nvidia", "Acción", "Estados Unidos", "Tecnología"),
    "GOOGL": ("Alphabet (Google)", "Acción", "Estados Unidos", "Tecnología"),
    "AMZN": ("Amazon", "Acción", "Estados Unidos", "Consumo"),
    "META": ("Meta (Facebook)", "Acción", "Estados Unidos", "Tecnología"),
    "TSLA": ("Tesla", "Acción", "Estados Unidos", "Automoción"),
    "BRK-B": ("Berkshire Hathaway", "Acción", "Estados Unidos", "Finanzas"),
    "AVGO": ("Broadcom", "Acción", "Estados Unidos", "Tecnología"),
    "LLY": ("Eli Lilly", "Acción", "Estados Unidos", "Salud"),
    "JPM": ("JPMorgan Chase", "Acción", "Estados Unidos", "Banca"),
    "V": ("Visa", "Acción", "Estados Unidos", "Finanzas"),
    "MA": ("Mastercard", "Acción", "Estados Unidos", "Finanzas"),
    "UNH": ("UnitedHealth", "Acción", "Estados Unidos", "Salud"),
    "XOM": ("ExxonMobil", "Acción", "Estados Unidos", "Energía"),
    "CVX": ("Chevron", "Acción", "Estados Unidos", "Energía"),
    "WMT": ("Walmart", "Acción", "Estados Unidos", "Consumo"),
    "COST": ("Costco", "Acción", "Estados Unidos", "Consumo"),
    "HD": ("Home Depot", "Acción", "Estados Unidos", "Consumo"),
    "PG": ("Procter & Gamble", "Acción", "Estados Unidos", "Consumo"),
    "JNJ": ("Johnson & Johnson", "Acción", "Estados Unidos", "Salud"),
    "ABBV": ("AbbVie", "Acción", "Estados Unidos", "Salud"),
    "MRK": ("Merck", "Acción", "Estados Unidos", "Salud"),
    "PFE": ("Pfizer", "Acción", "Estados Unidos", "Salud"),
    "ABT": ("Abbott", "Acción", "Estados Unidos", "Salud"),
    "TMO": ("Thermo Fisher", "Acción", "Estados Unidos", "Salud"),
    "AMGN": ("Amgen", "Acción", "Estados Unidos", "Salud"),
    "GILD": ("Gilead", "Acción", "Estados Unidos", "Salud"),
    "BMY": ("Bristol-Myers Squibb", "Acción", "Estados Unidos", "Salud"),
    "ISRG": ("Intuitive Surgical", "Acción", "Estados Unidos", "Salud"),
    "KO": ("Coca-Cola", "Acción", "Estados Unidos", "Consumo"),
    "PEP": ("PepsiCo", "Acción", "Estados Unidos", "Consumo"),
    "MCD": ("McDonald's", "Acción", "Estados Unidos", "Consumo"),
    "SBUX": ("Starbucks", "Acción", "Estados Unidos", "Consumo"),
    "NKE": ("Nike", "Acción", "Estados Unidos", "Consumo"),
    "DIS": ("Disney", "Acción", "Estados Unidos", "Medios"),
    "NFLX": ("Netflix", "Acción", "Estados Unidos", "Medios"),
    "CMCSA": ("Comcast", "Acción", "Estados Unidos", "Medios"),
    "BAC": ("Bank of America", "Acción", "Estados Unidos", "Banca"),
    "WFC": ("Wells Fargo", "Acción", "Estados Unidos", "Banca"),
    "C": ("Citigroup", "Acción", "Estados Unidos", "Banca"),
    "GS": ("Goldman Sachs", "Acción", "Estados Unidos", "Banca"),
    "MS": ("Morgan Stanley", "Acción", "Estados Unidos", "Banca"),
    "BLK": ("BlackRock", "Acción", "Estados Unidos", "Finanzas"),
    "AXP": ("American Express", "Acción", "Estados Unidos", "Finanzas"),
    "PYPL": ("PayPal", "Acción", "Estados Unidos", "Finanzas"),
    "SCHW": ("Charles Schwab", "Acción", "Estados Unidos", "Finanzas"),
    "ORCL": ("Oracle", "Acción", "Estados Unidos", "Tecnología"),
    "CRM": ("Salesforce", "Acción", "Estados Unidos", "Tecnología"),
    "ADBE": ("Adobe", "Acción", "Estados Unidos", "Tecnología"),
    "CSCO": ("Cisco", "Acción", "Estados Unidos", "Tecnología"),
    "IBM": ("IBM", "Acción", "Estados Unidos", "Tecnología"),
    "INTC": ("Intel", "Acción", "Estados Unidos", "Tecnología"),
    "AMD": ("AMD", "Acción", "Estados Unidos", "Tecnología"),
    "QCOM": ("Qualcomm", "Acción", "Estados Unidos", "Tecnología"),
    "TXN": ("Texas Instruments", "Acción", "Estados Unidos", "Tecnología"),
    "MU": ("Micron", "Acción", "Estados Unidos", "Tecnología"),
    "AMAT": ("Applied Materials", "Acción", "Estados Unidos", "Tecnología"),
    "NOW": ("ServiceNow", "Acción", "Estados Unidos", "Tecnología"),
    "INTU": ("Intuit", "Acción", "Estados Unidos", "Tecnología"),
    "PLTR": ("Palantir", "Acción", "Estados Unidos", "Tecnología"),
    "UBER": ("Uber", "Acción", "Estados Unidos", "Tecnología"),
    "ABNB": ("Airbnb", "Acción", "Estados Unidos", "Consumo"),
    "SNOW": ("Snowflake", "Acción", "Estados Unidos", "Tecnología"),
    "CRWD": ("CrowdStrike", "Acción", "Estados Unidos", "Tecnología"),
    "PANW": ("Palo Alto Networks", "Acción", "Estados Unidos", "Tecnología"),
    "COIN": ("Coinbase", "Acción", "Estados Unidos", "Finanzas"),
    "BA": ("Boeing", "Acción", "Estados Unidos", "Industria"),
    "CAT": ("Caterpillar", "Acción", "Estados Unidos", "Industria"),
    "GE": ("GE Aerospace", "Acción", "Estados Unidos", "Industria"),
    "HON": ("Honeywell", "Acción", "Estados Unidos", "Industria"),
    "LMT": ("Lockheed Martin", "Acción", "Estados Unidos", "Defensa"),
    "RTX": ("RTX (Raytheon)", "Acción", "Estados Unidos", "Defensa"),
    "UPS": ("UPS", "Acción", "Estados Unidos", "Industria"),
    "DE": ("John Deere", "Acción", "Estados Unidos", "Industria"),
    "F": ("Ford", "Acción", "Estados Unidos", "Automoción"),
    "GM": ("General Motors", "Acción", "Estados Unidos", "Automoción"),
    "T": ("AT&T", "Acción", "Estados Unidos", "Telecomunicaciones"),
    "VZ": ("Verizon", "Acción", "Estados Unidos", "Telecomunicaciones"),
    "NEE": ("NextEra Energy", "Acción", "Estados Unidos", "Energía"),
    "DUK": ("Duke Energy", "Acción", "Estados Unidos", "Energía"),
    "LIN": ("Linde", "Acción", "Estados Unidos", "Materiales"),
    "NEM": ("Newmont", "Acción", "Estados Unidos", "Materiales"),
    "FCX": ("Freeport-McMoRan", "Acción", "Estados Unidos", "Materiales"),
    "SPCX": ("SpaceX", "Acción", "Estados Unidos", "Industria"),
    "RIVN": ("Rivian", "Acción", "Estados Unidos", "Automoción"),
    "LCID": ("Lucid", "Acción", "Estados Unidos", "Automoción"),
    "SOFI": ("SoFi", "Acción", "Estados Unidos", "Finanzas"),
    "PLUG": ("Plug Power", "Acción", "Estados Unidos", "Energía"),
    "GME": ("GameStop", "Acción", "Estados Unidos", "Consumo"),
    "AMC": ("AMC Entertainment", "Acción", "Estados Unidos", "Medios"),
    "HOOD": ("Robinhood", "Acción", "Estados Unidos", "Finanzas"),
    "RBLX": ("Roblox", "Acción", "Estados Unidos", "Medios"),
    "DKNG": ("DraftKings", "Acción", "Estados Unidos", "Consumo"),
    "ROKU": ("Roku", "Acción", "Estados Unidos", "Medios"),
    "ETSY": ("Etsy", "Acción", "Estados Unidos", "Consumo"),
    "CHWY": ("Chewy", "Acción", "Estados Unidos", "Consumo"),
    "MARA": ("MARA Holdings", "Acción", "Estados Unidos", "Finanzas"),
    "IONQ": ("IonQ", "Acción", "Estados Unidos", "Tecnología"),
    "RKLB": ("Rocket Lab", "Acción", "Estados Unidos", "Industria"),
    # --- Acciones: España ---
    "SAN.MC": ("Banco Santander", "Acción", "España", "Banca"),
    "BBVA.MC": ("BBVA", "Acción", "España", "Banca"),
    "CABK.MC": ("CaixaBank", "Acción", "España", "Banca"),
    "SAB.MC": ("Banco Sabadell", "Acción", "España", "Banca"),
    "BKT.MC": ("Bankinter", "Acción", "España", "Banca"),
    "UNI.MC": ("Unicaja", "Acción", "España", "Banca"),
    "ITX.MC": ("Inditex", "Acción", "España", "Consumo"),
    "IBE.MC": ("Iberdrola", "Acción", "España", "Energía"),
    "ELE.MC": ("Endesa", "Acción", "España", "Energía"),
    "NTGY.MC": ("Naturgy", "Acción", "España", "Energía"),
    "REP.MC": ("Repsol", "Acción", "España", "Energía"),
    "RED.MC": ("Redeia (Red Eléctrica)", "Acción", "España", "Energía"),
    "ENG.MC": ("Enagás", "Acción", "España", "Energía"),
    "SLR.MC": ("Solaria", "Acción", "España", "Energía"),
    "ANE.MC": ("Acciona Energía", "Acción", "España", "Energía"),
    "TEF.MC": ("Telefónica", "Acción", "España", "Telecomunicaciones"),
    "CLNX.MC": ("Cellnex", "Acción", "España", "Telecomunicaciones"),
    "FER.MC": ("Ferrovial", "Acción", "España", "Industria"),
    "ACS.MC": ("ACS", "Acción", "España", "Industria"),
    "ANA.MC": ("Acciona", "Acción", "España", "Industria"),
    "SCYR.MC": ("Sacyr", "Acción", "España", "Industria"),
    "AENA.MC": ("Aena", "Acción", "España", "Industria"),
    "IAG.MC": ("IAG (Iberia, British Airways)", "Acción", "España", "Industria"),
    "AMS.MC": ("Amadeus", "Acción", "España", "Tecnología"),
    "IDR.MC": ("Indra", "Acción", "España", "Defensa"),
    "MAP.MC": ("Mapfre", "Acción", "España", "Finanzas"),
    "GRF.MC": ("Grifols", "Acción", "España", "Salud"),
    "ROVI.MC": ("Laboratorios Rovi", "Acción", "España", "Salud"),
    "ALM.MC": ("Almirall", "Acción", "España", "Salud"),
    "MTS.MC": ("ArcelorMittal", "Acción", "España", "Materiales"),
    "ACX.MC": ("Acerinox", "Acción", "España", "Materiales"),
    "COL.MC": ("Inmobiliaria Colonial", "Acción", "España", "Inmobiliaria"),
    "MRL.MC": ("Merlin Properties", "Acción", "España", "Inmobiliaria"),
    "LOG.MC": ("Logista", "Acción", "España", "Industria"),
    "FDR.MC": ("Fluidra", "Acción", "España", "Industria"),
    "PUIG.MC": ("Puig", "Acción", "España", "Consumo"),
    "MEL.MC": ("Meliá Hotels", "Acción", "España", "Consumo"),
    "VIS.MC": ("Viscofan", "Acción", "España", "Consumo"),
    "EBRO.MC": ("Ebro Foods", "Acción", "España", "Consumo"),
    "VID.MC": ("Vidrala", "Acción", "España", "Materiales"),
    "PSG.MC": ("Prosegur", "Acción", "España", "Industria"),
    "A3M.MC": ("Atresmedia", "Acción", "España", "Medios"),
    "GEST.MC": ("Gestamp", "Acción", "España", "Automoción"),
    "CIE.MC": ("CIE Automotive", "Acción", "España", "Automoción"),
    "ENC.MC": ("Ence", "Acción", "España", "Materiales"),
    "TUB.MC": ("Tubacex", "Acción", "España", "Materiales"),
    "FAE.MC": ("Faes Farma", "Acción", "España", "Salud"),
    "TRE.MC": ("Técnicas Reunidas", "Acción", "España", "Energía"),
    # --- Acciones: Europa ---
    "SPOT": ("Spotify", "Acción", "Suecia", "Medios"),
    "ASML.AS": ("ASML", "Acción", "Países Bajos", "Tecnología"),
    "INGA.AS": ("ING", "Acción", "Países Bajos", "Banca"),
    "AD.AS": ("Ahold Delhaize", "Acción", "Países Bajos", "Consumo"),
    "PRX.AS": ("Prosus", "Acción", "Países Bajos", "Tecnología"),
    "HEIA.AS": ("Heineken", "Acción", "Países Bajos", "Consumo"),
    "PHIA.AS": ("Philips", "Acción", "Países Bajos", "Salud"),
    "MC.PA": ("LVMH", "Acción", "Francia", "Consumo"),
    "OR.PA": ("L'Oréal", "Acción", "Francia", "Consumo"),
    "RMS.PA": ("Hermès", "Acción", "Francia", "Consumo"),
    "KER.PA": ("Kering", "Acción", "Francia", "Consumo"),
    "TTE.PA": ("TotalEnergies", "Acción", "Francia", "Energía"),
    "AIR.PA": ("Airbus", "Acción", "Francia", "Industria"),
    "SAF.PA": ("Safran", "Acción", "Francia", "Defensa"),
    "BNP.PA": ("BNP Paribas", "Acción", "Francia", "Banca"),
    "GLE.PA": ("Société Générale", "Acción", "Francia", "Banca"),
    "ACA.PA": ("Crédit Agricole", "Acción", "Francia", "Banca"),
    "CS.PA": ("AXA", "Acción", "Francia", "Finanzas"),
    "SAN.PA": ("Sanofi", "Acción", "Francia", "Salud"),
    "SU.PA": ("Schneider Electric", "Acción", "Francia", "Industria"),
    "AI.PA": ("Air Liquide", "Acción", "Francia", "Materiales"),
    "DG.PA": ("Vinci", "Acción", "Francia", "Industria"),
    "BN.PA": ("Danone", "Acción", "Francia", "Consumo"),
    "EL.PA": ("EssilorLuxottica", "Acción", "Francia", "Salud"),
    "RNO.PA": ("Renault", "Acción", "Francia", "Automoción"),
    "STLAM.MI": ("Stellantis", "Acción", "Italia", "Automoción"),
    "RACE.MI": ("Ferrari", "Acción", "Italia", "Automoción"),
    "ENEL.MI": ("Enel", "Acción", "Italia", "Energía"),
    "ENI.MI": ("Eni", "Acción", "Italia", "Energía"),
    "ISP.MI": ("Intesa Sanpaolo", "Acción", "Italia", "Banca"),
    "UCG.MI": ("UniCredit", "Acción", "Italia", "Banca"),
    "G.MI": ("Generali", "Acción", "Italia", "Finanzas"),
    "SAP.DE": ("SAP", "Acción", "Alemania", "Tecnología"),
    "SIE.DE": ("Siemens", "Acción", "Alemania", "Industria"),
    "ALV.DE": ("Allianz", "Acción", "Alemania", "Finanzas"),
    "MUV2.DE": ("Munich Re", "Acción", "Alemania", "Finanzas"),
    "DTE.DE": ("Deutsche Telekom", "Acción", "Alemania", "Telecomunicaciones"),
    "VOW3.DE": ("Volkswagen", "Acción", "Alemania", "Automoción"),
    "BMW.DE": ("BMW", "Acción", "Alemania", "Automoción"),
    "MBG.DE": ("Mercedes-Benz", "Acción", "Alemania", "Automoción"),
    "P911.DE": ("Porsche", "Acción", "Alemania", "Automoción"),
    "BAS.DE": ("BASF", "Acción", "Alemania", "Materiales"),
    "BAYN.DE": ("Bayer", "Acción", "Alemania", "Salud"),
    "ADS.DE": ("Adidas", "Acción", "Alemania", "Consumo"),
    "DBK.DE": ("Deutsche Bank", "Acción", "Alemania", "Banca"),
    "IFX.DE": ("Infineon", "Acción", "Alemania", "Tecnología"),
    "RHM.DE": ("Rheinmetall", "Acción", "Alemania", "Defensa"),
    "DHL.DE": ("DHL Group", "Acción", "Alemania", "Industria"),
    "NESN.SW": ("Nestlé", "Acción", "Suiza", "Consumo"),
    "NOVN.SW": ("Novartis", "Acción", "Suiza", "Salud"),
    "RO.SW": ("Roche", "Acción", "Suiza", "Salud"),
    "UBSG.SW": ("UBS", "Acción", "Suiza", "Banca"),
    "ZURN.SW": ("Zurich Insurance", "Acción", "Suiza", "Finanzas"),
    "CFR.SW": ("Richemont", "Acción", "Suiza", "Consumo"),
    "ABBN.SW": ("ABB", "Acción", "Suiza", "Industria"),
    "NOVO-B.CO": ("Novo Nordisk", "Acción", "Dinamarca", "Salud"),
    "MAERSK-B.CO": ("Maersk", "Acción", "Dinamarca", "Industria"),
    "VWS.CO": ("Vestas", "Acción", "Dinamarca", "Energía"),
    "ERIC-B.ST": ("Ericsson", "Acción", "Suecia", "Tecnología"),
    "VOLV-B.ST": ("Volvo", "Acción", "Suecia", "Industria"),
    "NOKIA.HE": ("Nokia", "Acción", "Finlandia", "Tecnología"),
    "EQNR.OL": ("Equinor", "Acción", "Noruega", "Energía"),
    "SHEL.L": ("Shell", "Acción", "Reino Unido", "Energía"),
    "BP.L": ("BP", "Acción", "Reino Unido", "Energía"),
    "AZN.L": ("AstraZeneca", "Acción", "Reino Unido", "Salud"),
    "GSK.L": ("GSK", "Acción", "Reino Unido", "Salud"),
    "HSBA.L": ("HSBC", "Acción", "Reino Unido", "Banca"),
    "BARC.L": ("Barclays", "Acción", "Reino Unido", "Banca"),
    "LLOY.L": ("Lloyds", "Acción", "Reino Unido", "Banca"),
    "ULVR.L": ("Unilever", "Acción", "Reino Unido", "Consumo"),
    "DGE.L": ("Diageo", "Acción", "Reino Unido", "Consumo"),
    "RIO.L": ("Rio Tinto", "Acción", "Reino Unido", "Materiales"),
    "GLEN.L": ("Glencore", "Acción", "Reino Unido", "Materiales"),
    "VOD.L": ("Vodafone", "Acción", "Reino Unido", "Telecomunicaciones"),
    "RR.L": ("Rolls-Royce", "Acción", "Reino Unido", "Industria"),
    "BA.L": ("BAE Systems", "Acción", "Reino Unido", "Defensa"),
    # --- Acciones: Asia y Oceanía ---
    "7203.T": ("Toyota", "Acción", "Japón", "Automoción"),
    "7267.T": ("Honda", "Acción", "Japón", "Automoción"),
    "6758.T": ("Sony", "Acción", "Japón", "Tecnología"),
    "7974.T": ("Nintendo", "Acción", "Japón", "Medios"),
    "9984.T": ("SoftBank", "Acción", "Japón", "Tecnología"),
    "6861.T": ("Keyence", "Acción", "Japón", "Tecnología"),
    "8306.T": ("Mitsubishi UFJ", "Acción", "Japón", "Banca"),
    "9983.T": ("Fast Retailing (Uniqlo)", "Acción", "Japón", "Consumo"),
    "TSM": ("TSMC", "Acción", "Taiwán", "Tecnología"),
    "005930.KS": ("Samsung Electronics", "Acción", "Corea del Sur", "Tecnología"),
    "000660.KS": ("SK Hynix", "Acción", "Corea del Sur", "Tecnología"),
    "005380.KS": ("Hyundai", "Acción", "Corea del Sur", "Automoción"),
    "0700.HK": ("Tencent", "Acción", "China", "Tecnología"),
    "BABA": ("Alibaba", "Acción", "China", "Consumo"),
    "JD": ("JD.com", "Acción", "China", "Consumo"),
    "PDD": ("PDD (Temu)", "Acción", "China", "Consumo"),
    "BIDU": ("Baidu", "Acción", "China", "Tecnología"),
    "1211.HK": ("BYD", "Acción", "China", "Automoción"),
    "NIO": ("NIO", "Acción", "China", "Automoción"),
    "3690.HK": ("Meituan", "Acción", "China", "Consumo"),
    "1810.HK": ("Xiaomi", "Acción", "China", "Tecnología"),
    "RELIANCE.NS": ("Reliance Industries", "Acción", "India", "Energía"),
    "TCS.NS": ("Tata Consultancy", "Acción", "India", "Tecnología"),
    "INFY": ("Infosys", "Acción", "India", "Tecnología"),
    "HDB": ("HDFC Bank", "Acción", "India", "Banca"),
    "BHP.AX": ("BHP", "Acción", "Australia", "Materiales"),
    "CBA.AX": ("Commonwealth Bank", "Acción", "Australia", "Banca"),
    # --- Acciones: América ---
    "SHOP": ("Shopify", "Acción", "Canadá", "Tecnología"),
    "MELI": ("MercadoLibre", "Acción", "Argentina", "Consumo"),
    "YPF": ("YPF", "Acción", "Argentina", "Energía"),
    "GGAL": ("Grupo Galicia", "Acción", "Argentina", "Banca"),
    "PETR4.SA": ("Petrobras", "Acción", "Brasil", "Energía"),
    "VALE3.SA": ("Vale", "Acción", "Brasil", "Materiales"),
    "ITUB4.SA": ("Itaú Unibanco", "Acción", "Brasil", "Banca"),
    "NU": ("Nu Holdings (Nubank)", "Acción", "Brasil", "Finanzas"),
    "WALMEX.MX": ("Walmart de México", "Acción", "México", "Consumo"),
    "AMXB.MX": ("América Móvil", "Acción", "México", "Telecomunicaciones"),
    "FEMSAUBD.MX": ("FEMSA", "Acción", "México", "Consumo"),
    "CEMEXCPO.MX": ("Cemex", "Acción", "México", "Materiales"),
    "SQM": ("SQM", "Acción", "Chile", "Materiales"),
    "SHOP.TO": ("Shopify (Toronto)", "Acción", "Canadá", "Tecnología"),
    "RY.TO": ("Royal Bank of Canada", "Acción", "Canadá", "Banca"),
    "ENB.TO": ("Enbridge", "Acción", "Canadá", "Energía"),
    # --- Acciones: ampliación (cuántica, IA, espacio, nuclear, internet, consumo, España, Europa, Asia) ---
    "QBTS": ("D-Wave Quantum", "Acción", "Estados Unidos", "Tecnología"),
    "RGTI": ("Rigetti Computing", "Acción", "Estados Unidos", "Tecnología"),
    "QUBT": ("Quantum Computing Inc.", "Acción", "Estados Unidos", "Tecnología"),
    "ARM": ("Arm Holdings", "Acción", "Reino Unido", "Tecnología"),
    "MRVL": ("Marvell Technology", "Acción", "Estados Unidos", "Tecnología"),
    "SMCI": ("Super Micro Computer", "Acción", "Estados Unidos", "Tecnología"),
    "DELL": ("Dell", "Acción", "Estados Unidos", "Tecnología"),
    "LRCX": ("Lam Research", "Acción", "Estados Unidos", "Tecnología"),
    "KLAC": ("KLA", "Acción", "Estados Unidos", "Tecnología"),
    "SNPS": ("Synopsys", "Acción", "Estados Unidos", "Tecnología"),
    "DDOG": ("Datadog", "Acción", "Estados Unidos", "Tecnología"),
    "MDB": ("MongoDB", "Acción", "Estados Unidos", "Tecnología"),
    "NET": ("Cloudflare", "Acción", "Estados Unidos", "Tecnología"),
    "ZS": ("Zscaler", "Acción", "Estados Unidos", "Tecnología"),
    "SOUN": ("SoundHound AI", "Acción", "Estados Unidos", "Tecnología"),
    "BBAI": ("BigBear.ai", "Acción", "Estados Unidos", "Tecnología"),
    "TEM": ("Tempus AI", "Acción", "Estados Unidos", "Salud"),
    "ASTS": ("AST SpaceMobile", "Acción", "Estados Unidos", "Telecomunicaciones"),
    "LUNR": ("Intuitive Machines", "Acción", "Estados Unidos", "Industria"),
    "JOBY": ("Joby Aviation", "Acción", "Estados Unidos", "Industria"),
    "ACHR": ("Archer Aviation", "Acción", "Estados Unidos", "Industria"),
    "OKLO": ("Oklo", "Acción", "Estados Unidos", "Energía"),
    "SMR": ("NuScale Power", "Acción", "Estados Unidos", "Energía"),
    "CEG": ("Constellation Energy", "Acción", "Estados Unidos", "Energía"),
    "VST": ("Vistra", "Acción", "Estados Unidos", "Energía"),
    "FSLR": ("First Solar", "Acción", "Estados Unidos", "Energía"),
    "ENPH": ("Enphase Energy", "Acción", "Estados Unidos", "Energía"),
    "MSTR": ("Strategy (MicroStrategy)", "Acción", "Estados Unidos", "Tecnología"),
    "XYZ": ("Block (Square)", "Acción", "Estados Unidos", "Finanzas"),
    "AFRM": ("Affirm", "Acción", "Estados Unidos", "Finanzas"),
    "UPST": ("Upstart", "Acción", "Estados Unidos", "Finanzas"),
    "RDDT": ("Reddit", "Acción", "Estados Unidos", "Medios"),
    "SNAP": ("Snap", "Acción", "Estados Unidos", "Medios"),
    "PINS": ("Pinterest", "Acción", "Estados Unidos", "Medios"),
    "DUOL": ("Duolingo", "Acción", "Estados Unidos", "Tecnología"),
    "ZM": ("Zoom", "Acción", "Estados Unidos", "Tecnología"),
    "DASH": ("DoorDash", "Acción", "Estados Unidos", "Consumo"),
    "LYFT": ("Lyft", "Acción", "Estados Unidos", "Tecnología"),
    "BKNG": ("Booking", "Acción", "Estados Unidos", "Consumo"),
    "EXPE": ("Expedia", "Acción", "Estados Unidos", "Consumo"),
    "MAR": ("Marriott", "Acción", "Estados Unidos", "Consumo"),
    "DAL": ("Delta Air Lines", "Acción", "Estados Unidos", "Industria"),
    "UAL": ("United Airlines", "Acción", "Estados Unidos", "Industria"),
    "CCL": ("Carnival", "Acción", "Estados Unidos", "Consumo"),
    "RCL": ("Royal Caribbean", "Acción", "Estados Unidos", "Consumo"),
    "CVNA": ("Carvana", "Acción", "Estados Unidos", "Consumo"),
    "TGT": ("Target", "Acción", "Estados Unidos", "Consumo"),
    "LOW": ("Lowe's", "Acción", "Estados Unidos", "Consumo"),
    "LULU": ("Lululemon", "Acción", "Estados Unidos", "Consumo"),
    "CMG": ("Chipotle", "Acción", "Estados Unidos", "Consumo"),
    "MNST": ("Monster Beverage", "Acción", "Estados Unidos", "Consumo"),
    "CELH": ("Celsius", "Acción", "Estados Unidos", "Consumo"),
    "PM": ("Philip Morris", "Acción", "Estados Unidos", "Consumo"),
    "MDLZ": ("Mondelez", "Acción", "Estados Unidos", "Consumo"),
    "HSY": ("Hershey", "Acción", "Estados Unidos", "Consumo"),
    "EL": ("Estée Lauder", "Acción", "Estados Unidos", "Consumo"),
    "HIMS": ("Hims & Hers", "Acción", "Estados Unidos", "Salud"),
    "MRNA": ("Moderna", "Acción", "Estados Unidos", "Salud"),
    "BNTX": ("BioNTech", "Acción", "Alemania", "Salud"),
    "VRTX": ("Vertex", "Acción", "Estados Unidos", "Salud"),
    "REGN": ("Regeneron", "Acción", "Estados Unidos", "Salud"),
    "NVO": ("Novo Nordisk (EE. UU.)", "Acción", "Dinamarca", "Salud"),
    "GRE.MC": ("Grenergy", "Acción", "España", "Energía"),
    "ADX.MC": ("Audax Renovables", "Acción", "España", "Energía"),
    "OHLA.MC": ("OHLA", "Acción", "España", "Industria"),
    "NEA.MC": ("Nicolás Correa", "Acción", "España", "Industria"),
    "ECR.MC": ("Ercros", "Acción", "España", "Materiales"),
    "DIA.MC": ("DIA", "Acción", "España", "Consumo"),
    "R4.MC": ("Renta 4", "Acción", "España", "Finanzas"),
    "LDA.MC": ("Línea Directa", "Acción", "España", "Finanzas"),
    "CAF.MC": ("CAF", "Acción", "España", "Industria"),
    "HOME.MC": ("Neinor Homes", "Acción", "España", "Inmobiliaria"),
    "ADYEN.AS": ("Adyen", "Acción", "Países Bajos", "Finanzas"),
    "UMG.AS": ("Universal Music", "Acción", "Países Bajos", "Medios"),
    "RACE": ("Ferrari (EE. UU.)", "Acción", "Italia", "Automoción"),
    "LDO.MI": ("Leonardo", "Acción", "Italia", "Defensa"),
    "SAAB-B.ST": ("Saab", "Acción", "Suecia", "Defensa"),
    "HO.PA": ("Thales", "Acción", "Francia", "Defensa"),
    "ORA.PA": ("Orange", "Acción", "Francia", "Telecomunicaciones"),
    "VIE.PA": ("Veolia", "Acción", "Francia", "Industria"),
    "ENR.DE": ("Siemens Energy", "Acción", "Alemania", "Energía"),
    "ZAL.DE": ("Zalando", "Acción", "Alemania", "Consumo"),
    "DTG.DE": ("Daimler Truck", "Acción", "Alemania", "Automoción"),
    "RYAAY": ("Ryanair", "Acción", "Irlanda", "Industria"),
    "CRH": ("CRH", "Acción", "Irlanda", "Materiales"),
    "TTWO": ("Take-Two Interactive", "Acción", "Estados Unidos", "Medios"),
    "U": ("Unity Software", "Acción", "Estados Unidos", "Tecnología"),
    "SE": ("Sea Limited", "Acción", "Singapur", "Consumo"),
    "GRAB": ("Grab", "Acción", "Singapur", "Tecnología"),
    "CPNG": ("Coupang", "Acción", "Corea del Sur", "Consumo"),
    "TCEHY": ("Tencent (EE. UU.)", "Acción", "China", "Tecnología"),
    "XPEV": ("XPeng", "Acción", "China", "Automoción"),
    "LI": ("Li Auto", "Acción", "China", "Automoción"),
    "9988.HK": ("Alibaba (Hong Kong)", "Acción", "China", "Consumo"),
    "6501.T": ("Hitachi", "Acción", "Japón", "Industria"),
    "8035.T": ("Tokyo Electron", "Acción", "Japón", "Tecnología"),
    # --- Empresas pequeñas (fondo que las agrupa) ---
    "IWM": ("Empresas pequeñas de EE. UU. (ETF Russell 2000)", "ETF", "Estados Unidos", "—"),
    # --- Divisas ---
    "EURUSD=X": ("Euro / Dólar", "Divisa", "—", "—"),
    "GBPUSD=X": ("Libra / Dólar", "Divisa", "—", "—"),
    "USDJPY=X": ("Dólar / Yen", "Divisa", "—", "—"),
    "EURGBP=X": ("Euro / Libra", "Divisa", "—", "—"),
    "EURJPY=X": ("Euro / Yen", "Divisa", "—", "—"),
    "EURCHF=X": ("Euro / Franco suizo", "Divisa", "—", "—"),
    "USDCHF=X": ("Dólar / Franco suizo", "Divisa", "—", "—"),
    "AUDUSD=X": ("Dólar australiano / Dólar", "Divisa", "—", "—"),
    "USDCAD=X": ("Dólar / Dólar canadiense", "Divisa", "—", "—"),
    "USDMXN=X": ("Dólar / Peso mexicano", "Divisa", "—", "—"),
    "USDBRL=X": ("Dólar / Real brasileño", "Divisa", "—", "—"),
    "USDCNY=X": ("Dólar / Yuan chino", "Divisa", "—", "—"),
    # --- Materias primas ---
    "GC=F": ("Oro", "Materia prima", "—", "—"),
    "SI=F": ("Plata", "Materia prima", "—", "—"),
    "BZ=F": ("Petróleo Brent", "Materia prima", "—", "—"),
    "CL=F": ("Petróleo WTI", "Materia prima", "—", "—"),
    "NG=F": ("Gas natural", "Materia prima", "—", "—"),
    "HG=F": ("Cobre", "Materia prima", "—", "—"),
    "ZW=F": ("Trigo", "Materia prima", "—", "—"),
    "ZC=F": ("Maíz", "Materia prima", "—", "—"),
    "KC=F": ("Café", "Materia prima", "—", "—"),
    # --- Criptomonedas ---
    "BTC-USD": ("Bitcoin", "Criptomoneda", "—", "—"),
    "ETH-USD": ("Ethereum", "Criptomoneda", "—", "—"),
    "SOL-USD": ("Solana", "Criptomoneda", "—", "—"),
}

for _t, (_pais, _nombre, _) in INDICES.items():
    ACTIVOS.setdefault(_t, (_nombre, "Índice", _pais, "—"))

# Marcas o nombres populares que no coinciden con el de la empresa cotizada.
ALIAS = {
    "zara": "ITX.MC", "bershka": "ITX.MC", "pull and bear": "ITX.MC", "massimo dutti": "ITX.MC",
    "iberia": "IAG.MC", "british airways": "IAG.MC", "vueling": "IAG.MC", "red electrica": "RED.MC",
    "google": "GOOGL", "youtube": "GOOGL", "facebook": "META", "instagram": "META", "whatsapp": "META",
    "temu": "PDD", "uniqlo": "9983.T", "playstation": "6758.T", "mario": "7974.T", "starlink": "SPCX",
    "chatgpt": "MSFT", "dwave": "QBTS", "d wave": "QBTS", "square": "XYZ", "microstrategy": "MSTR", "iphone": "AAPL", "windows": "MSFT", "nubank": "NU", "mercado libre": "MELI",
    "ibex": "^IBEX", "sp500": "^GSPC", "s&p": "^GSPC", "dow": "^DJI", "nasdaq": "^IXIC", "dax": "^GDAXI",
    "nikkei": "^N225", "euro stoxx": "^STOXX50E", "petroleo": "BZ=F", "dolar": "EURUSD=X",
}

ORDEN_TIPOS = ["Índice", "Acción", "ETF", "Divisa", "Materia prima", "Criptomoneda"]


def _normal(texto: str) -> str:
    """Minúsculas y sin acentos, para buscar «japon» o «telefonica»."""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_acentos.lower().strip()


def nombre(ticker: str, nombre_alternativo: str | None = None) -> str:
    """Nombre legible: «Amazon», «IBEX 35 (España)», «Euro / Dólar»."""
    if ticker in ACTIVOS:
        n, tipo, pais, _ = ACTIVOS[ticker]
        return f"{n} ({pais})" if tipo == "Índice" else n
    return nombre_alternativo or ticker


def descripcion(ticker: str) -> str:
    """Texto de la lista desplegable: «Amazon · Acción · Estados Unidos»."""
    if ticker not in ACTIVOS:
        return ticker
    n, tipo, pais, sector = ACTIVOS[ticker]
    partes = [n, tipo if tipo != "Acción" else sector] + ([pais] if pais != "—" else [])
    return " · ".join(partes)


def _sin_signos(texto: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", texto)


def buscar(texto: str) -> list[str]:
    """Tickers del catálogo que encajan con lo escrito (nombre, país, tipo, sector o ticker), mejores primero."""
    clave = _normal(texto).lstrip("^")
    if not clave:
        return []
    clave_limpia = _sin_signos(clave)
    alias = ALIAS.get(clave) or ALIAS.get(clave_limpia)
    exactos, empiezan, contienen = ([alias] if alias else []), [], []
    for t, (n, tipo, pais, sector) in ACTIVOS.items():
        nn, tt = _normal(n), _normal(t).lstrip("^")
        nl = _sin_signos(nn)
        if t == alias:
            continue
        if clave in (tt, nn, tt.split(".")[0]) or (clave_limpia and clave_limpia == nl):
            exactos.append(t)
        elif nn.startswith(clave) or clave in nn.split() or (clave_limpia and nl.startswith(clave_limpia)):
            empiezan.append(t)
        elif clave in nn or clave in (_normal(pais), _normal(tipo), _normal(sector)):
            contienen.append(t)
    por_tipo = lambda t: ORDEN_TIPOS.index(ACTIVOS[t][1])  # noqa: E731 - índices antes que acciones
    primeros = exactos[:1] if alias else []
    resto = [t for t in exactos if t not in primeros]
    return primeros + sorted(resto, key=por_tipo) + sorted(empiezan, key=por_tipo) + sorted(contienen, key=por_tipo)


def buscar_en_yahoo(texto: str, maximo: int = 6) -> list[tuple[str, str]]:
    """Busca por nombre en Yahoo Finance lo que no está en el catálogo: [(ticker, nombre), …]."""
    try:
        import yfinance as yf

        resultados = yf.Search(texto, max_results=maximo, news_count=0).quotes or []
    except Exception:
        return []
    prioridad = {"EQUITY": 0, "INDEX": 1, "ETF": 2, "CRYPTOCURRENCY": 3, "FUTURE": 4, "CURRENCY": 5}
    validos = [q for q in resultados if q.get("quoteType") in prioridad and q.get("symbol")]
    validos.sort(key=lambda q: prioridad[q["quoteType"]])  # sort estable: respeta la relevancia de Yahoo
    return [(q["symbol"], q.get("shortname") or q.get("longname") or q["symbol"]) for q in validos]


def sector(ticker: str) -> str:
    return ACTIVOS[ticker][3] if ticker in ACTIVOS else "—"


def tipo(ticker: str) -> str:
    return ACTIVOS[ticker][1] if ticker in ACTIVOS else "Otro"


def pais(ticker: str) -> str:
    return ACTIVOS[ticker][2] if ticker in ACTIVOS else "—"


def opciones_ordenadas() -> list[str]:
    """Tickers del catálogo ordenados por tipo y nombre (para la lista desplegable)."""
    return sorted(ACTIVOS, key=lambda t: (ORDEN_TIPOS.index(ACTIVOS[t][1]), ACTIVOS[t][2] != "Estados Unidos"
                                          if ACTIVOS[t][1] == "Índice" else 0, _normal(ACTIVOS[t][0])))
