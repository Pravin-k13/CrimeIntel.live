import os 
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "fir_texts.pkl"), "rb") as f:
    texts = pickle.load(f)

index = faiss.read_index(
    os.path.join(BASE_DIR, "fir_index_small.index")
)

df = pd.read_csv(
    os.path.join(BASE_DIR, "archive", "FIR_DATABASE.csv"),
    low_memory=False
)
df = df.fillna("")
# ✅ Normalize officer names for matching
df['IOName_clean'] = df['IOName'] \
    .str.replace(r'[^\w\s]', '', regex=True) \
    .str.lower() \
    .str.replace(r'\s+', ' ', regex=True)
df['FIR_YEAR'] = pd.to_numeric(df['FIR_YEAR'], errors='coerce')
# df['FIR_YEAR'] = df['FIR_YEAR'].str.strip()

df = df.reset_index(drop=True)
df['Case_No'] = df.index + 1

print(f"Loaded {len(df)} records from CSV")
print(f"Loaded {len(texts)} texts from pickle")

ALL_CRIME_TYPES = set()
if 'CrimeGroup_Name' in df.columns:
    ALL_CRIME_TYPES.update(df['CrimeGroup_Name'].unique())
if 'CrimeHead_Name' in df.columns:
    ALL_CRIME_TYPES.update(df['CrimeHead_Name'].unique())

ALL_CRIME_TYPES = {crime.strip() for crime in ALL_CRIME_TYPES if crime and str(crime).lower() not in ['nan', '', 'none']}

print(f"Found {len(ALL_CRIME_TYPES)} unique crime types in dataset")
print(f"Sample crime types: {list(ALL_CRIME_TYPES)[:10]}")


# Semantic embedding model
embed_model = SentenceTransformer(
    "intfloat/e5-small-v2"
)

class ChatHistory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.context = {}

    def add_interaction(self, query, facts, answer, filtered_df):
        interaction = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': query,
            'facts': facts,
            'answer': answer,
            'filtered_df': filtered_df,
            'entities': self.extract_entities_from_df(filtered_df)
        }
        self.history.append(interaction)

        if filtered_df is not None and len(filtered_df) > 0:
            if 'IOName' in filtered_df.columns and len(filtered_df['IOName'].unique()) > 0:
                self.context['last_officer'] = filtered_df['IOName'].iloc[0]
            if 'District_Name' in filtered_df.columns:
                self.context['last_district'] = filtered_df['District_Name'].iloc[0]
            if 'CrimeGroup_Name' in filtered_df.columns:
                self.context['last_crime_type'] = list(filtered_df['CrimeGroup_Name'].unique())
            year_match = re.search(r'\b(19|20)\d{2}\b', query)
            if year_match:
                self.context['last_year'] = year_match.group(0)
            else:
                self.context.pop('last_year', None)

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def extract_entities_from_df(self, df):
        entities = {}
        if df is not None and len(df) > 0:
            if 'IOName' in df.columns:
                entities['officer'] = df['IOName'].iloc[0]
            if 'District_Name' in df.columns:
                entities['district'] = df['District_Name'].iloc[0]
        return entities

    def get_last_interaction(self):
        return self.history[-1] if self.history else None

    def clear(self):
        self.history = []
        self.context = {}

chat_history = ChatHistory(max_history=10)

def find_crime_type_in_query(query):
    """
    Dynamically detect crime type from query by matching against actual dataset crime types
    """
    query_lower = query.lower()

    # ✅ FIX HERE (TOP)
    if "posco" in query_lower:
        query_lower = query_lower.replace("posco", "pocso")

    matched_crimes = []

    for crime_type in ALL_CRIME_TYPES:
        crime_lower = crime_type.lower()

        crime_word_pattern = r'\b' + re.escape(crime_lower) + r'\b'
        if re.search(crime_word_pattern, query_lower) or query_lower in crime_lower:
            matched_crimes.append(crime_type)
            continue

        crime_words = crime_lower.split()
        query_words = query_lower.split()

        significant_words = [w for w in crime_words if len(w) > 3]  
        matches = sum(1 for word in significant_words if word in query_words)

        if matches > 0 and matches >= len(significant_words) * 0.6:  
            matched_crimes.append(crime_type)

    crime_aliases = {
        'murder': ['MURDER', 'CULPABLE HOMICIDE', 'HOMICIDE'],
        'theft': ['THEFT', 'ROBBERY', 'BURGLARY', 'DACOITY'],
        'rape': ['RAPE', 'SEXUAL'],
        'assault': ['ASSAULT', 'HURT', 'GRIEVOUS HURT'],
        'kidnapping': ['KIDNAPPING', 'ABDUCTION'],
        'accident': ['ACCIDENT', 'MOTOR VEHICLE'],
        
        # ✅ FIXED
        'pocso': ['POCSO', 'PROTECTION OF CHILDREN'],

        'molestation': ['MOLESTATION', 'OUTRAGING MODESTY'],
        'missing': ['MISSING PERSON'],
        'riots': ['RIOTS', 'UNLAWFUL ASSEMBLY'],
        'dowry': ['DOWRY', 'CRUELTY BY HUSBAND'],
        'cheating': ['CHEATING', 'CRIMINAL BREACH OF TRUST'],
        'drug': ['NDPS', 'NARCOTIC', 'DRUG']
    }

    for keyword, aliases in crime_aliases.items():
        if keyword in query_lower:
            for alias in aliases:
                for crime_type in ALL_CRIME_TYPES:
                    if alias.lower() in crime_type.lower():
                        matched_crimes.append(crime_type)

    return list(set(matched_crimes))


def search_fir_details(query, k=20):
    query_embedding = embed_model.encode(
        [f"query: {query}"],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        k
    )
    return indices[0], distances[0]

def extract_query_entities(query, chat_context=None):
    query_lower = query.lower()
    entities = {
        'officer_name': None,
        'crime_type': None,
        'district': None,
        'year': None
    }
    # ✅ GENERIC DISTRICT/CITY DETECTION
    # Detect any district/city that exists in the CSV

    districts = (
        df['District_Name']
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    matched_districts = []

    for district in districts:

        district_lower = district.lower()

        # Exact phrase match
        if district_lower in query_lower:
            matched_districts.append(district)

    # Store detected district(s)
    if matched_districts:
        entities['district'] = matched_districts
    if "pocso" in query_lower or "posco" in query_lower:
        entities['crime_type'] = ["POCSO"]

    follow_up_patterns = [
        r'\b(he|she|him|her|they|them|his|her|their)\b',
        r'\b(this|that|the same)\s+(officer|person|investigator)\b',
        r'\bof\s+(them|these|those)\b'
    ]

    is_follow_up = any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
    
    # ✅ Only reuse officer, NOT crime
    # ✅ ONLY apply if user explicitly refers to officer
    if "his" in query_lower or "that officer" in query_lower:
        if chat_context and 'last_officer' in chat_context:
            officer_full = chat_context['last_officer']
            officer_name = re.sub(r'\s*\([^)]*\)', '', officer_full).strip()
            entities['officer_name'] = officer_name

    if is_follow_up and chat_context:
        if 'last_officer' in chat_context:
            officer_full = chat_context['last_officer']
            officer_name = re.sub(r'\s*\([^)]*\)', '', officer_full).strip()
            entities['officer_name'] = officer_name
        if 'year' in query_lower or 'when' in query_lower:
            entities['year'] = None
        else:
            if 'last_year' in chat_context:
                entities['year'] = chat_context['last_year']
            if 'last_crime_type' in chat_context and not any(word in query_lower for word in ['what', 'which', 'crime', 'case']):
                entities['crime_type'] = chat_context['last_crime_type']
    else:
        # =========================================================
        # OFFICER NAME EXTRACTION
        # =========================================================

        officer_patterns = [

            # How many cases did RAMESH H HANAPUR handle?
            r'how\s+many\s+cases?\s+did\s+(.+?)\s+handle\b',

            # How many cases has RAMESH H HANAPUR handled?
            r'how\s+many\s+cases?\s+has\s+(.+?)\s+handled\b',

            # How many cases does RAMESH H HANAPUR handle?
            r'how\s+many\s+cases?\s+does\s+(.+?)\s+handle\b',

            # RAMESH H HANAPUR handled how many cases?
            r'(.+?)\s+handled\s+how\s+many\s+cases?\b',

            # How many cases were handled by RAMESH?
            r'cases?\s+(?:were\s+)?handled\s+by\s+(.+?)(?:\?|$)',

            # Cases handled by RAMESH
            r'cases?\s+handled\s+by\s+(.+?)(?:\?|$)',

            # handled by RAMESH
            r'handled\s+by\s+officer\s+(.+?)(?:\?|$)',
            r'handled\s+by\s+(.+?)(?:\?|$)',

            # officer RAMESH
            r'officer\s+(.+?)(?:\?|$)',

            # RAMESH's cases
            r"(.+?)['’]s\s+cases?\b",

            # RAMESH handled
            r'(.+?)\s+handled\b',

            # who is RAMESH
            r'who\s+is\s+(.+?)(?:\?|$)',

            # tell me about RAMESH
            r'tell\s+me\s+about\s+(.+?)(?:\?|$)',
        ]

        for pattern in officer_patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE
            )

            if match:

                captured = match.group(1).strip()

                # Remove rank information such as:
                # RAMESH H HANAPUR (PSI)
                captured = re.sub(
                    r'\([^)]*\)',
                    '',
                    captured
                )

                # Remove unnecessary words
                captured = re.sub(
                    r'\b(case|cases|handled|handle|by|officer|did|has|have|does)\b',
                    '',
                    captured,
                    flags=re.IGNORECASE
                )

                # Normalize spaces
                captured = re.sub(
                    r'\s+',
                    ' ',
                    captured
                ).strip()

                if captured:
                    entities['officer_name'] = captured.lower()

                    print(
                        "✅ Detected Officer:",
                        entities['officer_name']
                    )

                    break

        for pattern in officer_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                captured = match.group(1).strip()
                
                # remove only rank like (PSI)
                captured = re.sub(r'\(.*?\)', '', captured)

                # remove unwanted words
                captured = re.sub(r'\b(case|cases|handled|by|officer)\b', '', captured, flags=re.IGNORECASE)

                captured = re.sub(r'\s+', ' ', captured).strip().lower()

                # ✅ REMOVE unwanted words
                captured = re.sub(r'\b(how many|cases|case|handled|by|officer)\b', '', captured, flags=re.IGNORECASE)

                captured = re.sub(r'\s+', ' ', captured).strip()
                captured = captured.lower()

                entities['officer_name'] = captured   # ✅ IMPORTANT

                break

    matched_crimes = find_crime_type_in_query(query)
    if matched_crimes:
        entities['crime_type'] = matched_crimes
        print(f" Detected crime types: {matched_crimes}")
    
    year_range = re.findall(r'\b(19\d{2}|20\d{2})\b', query)

    if len(year_range) >= 2:
        entities['year_range'] = year_range

    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    if year_match:
        entities['year'] = year_match.group(0)

    # ✅ Extract case number
    case_match = re.search(r'\bcase\s*(no|number)?\s*(\d+)\b', query.lower())

    if case_match:
        entities['case_no'] = int(case_match.group(2))
        print("✅ Detected Case No:", entities['case_no'])
    
    return entities

def filter_dataframe(df, entities, indices):
    relevant_df = df.iloc[indices].copy().reset_index(drop=True)


    if entities['officer_name']:
        mask = relevant_df['IOName'].str.contains(
            entities['officer_name'],
            case=False,
            na=False,
            regex=False
        )
        relevant_df = relevant_df[mask].reset_index(drop=True)

    if entities['crime_type']:
        mask = pd.Series(
            [False] * len(relevant_df),
            index=relevant_df.index
        )

        for crime in entities['crime_type']:
            keyword = crime.lower().strip()

            mask |= relevant_df['CrimeGroup_Name'].str.lower().str.contains(
                keyword, na=False
            )

            mask |= relevant_df['CrimeHead_Name'].str.lower().str.contains(
                keyword, na=False
            )

        relevant_df = relevant_df[mask]
    # ✅ GENERIC DISTRICT FILTER
    if entities['district']:

        relevant_df = relevant_df[
            relevant_df['District_Name'].isin(
                entities['district']
            )
        ]

        print(
            "✅ After district filter:",
            len(relevant_df)
        )

    if entities['year']:
        relevant_df = relevant_df[relevant_df['FIR_YEAR'] == entities['year']].reset_index(drop=True)

    return relevant_df

def generate_facts_from_records(filtered_df, entities, query):
    facts = []

    if len(filtered_df) == 0:
        return ["No matching records found."], {}

    stats = {}

    if 'IOName' in filtered_df.columns:
        officers = filtered_df['IOName'].unique()
        if len(officers) > 0:
            stats['num_cases'] = len(filtered_df)
            if entities.get('officer_name') and len(officers) == 1:
                stats['officer'] = officers[0]
            elif entities.get('officer_name'):
                stats['officer'] = officers[0]
            else:
                stats['officer'] = None
                stats['all_officers'] = filtered_df['IOName'].value_counts().head(5).to_dict()

    if 'District_Name' in filtered_df.columns:
        districts = filtered_df['District_Name'].unique()
        if len(districts) > 0:
            stats['districts'] = list(districts[:3])

    if 'UnitName' in filtered_df.columns:
        units = filtered_df['UnitName'].unique()
        units = [u for u in units if str(u) != 'nan' and str(u) != '']
        if len(units) > 0:
            stats['units'] = list(units[:2])

    if 'CrimeGroup_Name' in filtered_df.columns:
        crimes = filtered_df['CrimeGroup_Name'].value_counts().head(5)
        stats['crime_types'] = dict(crimes)

    if 'CrimeHead_Name' in filtered_df.columns:
        crime_heads = filtered_df['CrimeHead_Name'].value_counts().head(5)
        stats['crime_heads'] = dict(crime_heads)

    if 'FIR_YEAR' in filtered_df.columns:
        years = filtered_df['FIR_YEAR'].value_counts()
        stats['years'] = dict(years)

    if 'VICTIM COUNT' in filtered_df.columns:
        victim_counts = filtered_df['VICTIM COUNT'].astype(str).str.extract(r'(\d+)', expand=False)
        victim_counts = pd.to_numeric(victim_counts, errors='coerce')
        total_victims = victim_counts.sum()
        if not pd.isna(total_victims) and total_victims > 0:
            stats['total_victims'] = int(total_victims)

    if 'Accused Count' in filtered_df.columns:
        accused_counts = filtered_df['Accused Count'].astype(str).str.extract(r'(\d+)', expand=False)
        accused_counts = pd.to_numeric(accused_counts, errors='coerce')
        total_accused = accused_counts.sum()
        if not pd.isna(total_accused) and total_accused > 0:
            stats['total_accused'] = int(total_accused)

    if 'Arrested Count  No.' in filtered_df.columns:
        arrested_counts = filtered_df['Arrested Count No.'].astype(str).str.extract(r'(\d+)', expand=False)
        arrested_counts = pd.to_numeric(arrested_counts, errors='coerce')
        total_arrested = arrested_counts.sum()
        if not pd.isna(total_arrested) and total_arrested > 0:
            stats['total_arrested'] = int(total_arrested)

    return facts, stats

def summarize_case(filtered_df):
    if len(filtered_df) == 0:
        return "No case details found."

    total_cases = len(filtered_df)

    # Top crime
    top_crime = filtered_df['CrimeHead_Name'].value_counts().idxmax()

    # Top location
    top_location = filtered_df['Village_Area_Name'].value_counts().idxmax()

    # Officer summary
    top_officers = filtered_df['IOName'].value_counts().head(3)
    officer_list = ", ".join([o.split('(')[0].strip() for o in top_officers.index])

    # Arrest summary
    total_male = pd.to_numeric(filtered_df['Arrested Male'], errors='coerce').sum()
    total_female = pd.to_numeric(filtered_df['Arrested Female'], errors='coerce').sum()

    # Victim summary
    total_victims = pd.to_numeric(filtered_df['VICTIM COUNT'], errors='coerce').sum()

    # Accused summary
    total_accused = pd.to_numeric(filtered_df['Accused Count'], errors='coerce').sum()

    # FIR type
    fir_types = filtered_df['FIR Type'].value_counts().head(2).to_dict()

    # Complaint mode
    complaint_modes = filtered_df['Complaint_Mode'].value_counts().head(2).to_dict()

    return f"""
Summary of Theft Cases:

Total Cases: {total_cases}

Most Common Crime: {top_crime}
Most Affected Area: {top_location}

Top Officers Involved: {officer_list}

Victims: {int(total_victims)}
Accused: {int(total_accused)}

Arrests:
Male: {int(total_male)}
Female: {int(total_female)}

FIR Types: {fir_types}
Complaint Modes: {complaint_modes}
"""

def detailed_case_view(filtered_df):

    row = filtered_df.iloc[0]

    # ✅ get description safely
    description = row.get('Case_Description', '')

    short_summary = ""

    if description and str(description).strip() != "":

        # split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', str(description))

        # take first 2 important sentences
        short_summary = " ".join(sentences[:2]).strip()

    else:

        # fallback summary
        short_summary = (
            f"A case related to {row.get('CrimeGroup_Name')} "
            f"was reported near {row.get('Place of Offence')} "
            f"in {row.get('District_Name')} district."
        )

    return f"""
CASE SUMMARY:

{short_summary}

------------------------------------------------

Case No: {row.get('Case_No')}

District: {row.get('District_Name')}

Police Station: {row.get('UnitName')}

Officer: {row.get('IOName')}

Crime Type: {row.get('CrimeGroup_Name')}

Offence: {row.get('CrimeHead_Name')}

Act Section:
{row.get('ActSection')}

Location:
{row.get('Place of Offence')}

Area:
{row.get('Village_Area_Name')}

Distance from PS:
{row.get('Distance from PS')}

Victim Details:
Male: {row.get('Male')}
Female: {row.get('Female')}
Boy: {row.get('Boy')}
Girl: {row.get('Girl')}

Victim Count:
{row.get('VICTIM COUNT')}

Accused Count:
{row.get('Accused Count')}

Chargesheeted:
{row.get('Accused_ChargeSheeted Count')}

Convictions:
{row.get('Conviction Count')}

------------------------------------------------

Full Incident Description:

{description}
"""

def generate_conversational_response(stats, query, entities, chat_context, filtered_df):
    """Generate response using templates"""

    if not stats:
        return "There is no any relevant records matched for your query"

    query_lower = query.lower()
    num_cases = stats.get('num_cases', 0)
    officer_name = stats.get('officer')

    if officer_name:
        officer_name = officer_name.split('(')[0].strip()
    else:
        officer_name = None

    if re.search(r'who\s+is', query_lower):
        officer_name = stats.get('officer')

        if officer_name:
            officer_name = officer_name.split('(')[0].strip()
        else:
            officer_name = None
        num_cases = stats.get('num_cases', 0)
        districts = stats.get('districts', [])
        years = sorted(stats.get('years', {}).keys())
        year_range = f"from {years[0]} to {years[-1]}" if len(years) > 1 else years[0] if years else "unknown period"
        top_crimes = list(stats.get('crime_types', {}).keys())[:3]

        response = f"{officer_name} is a police officer in Karnataka, based in {', '.join(districts)}. "
        response += f"They have handled {num_cases} case(s) {year_range}."
        if top_crimes:
            response += f" Most frequent case types: {', '.join(top_crimes)}."
        if 'total_accused' in stats:
            response += f" Total accused: {stats['total_accused']}, arrested: {stats.get('total_arrested', 0)}."
        return response

    crime_name = None
    if entities.get('crime_type'):
        crime_name = ", ".join(entities['crime_type'][:3])
        if "theft" in query.lower():
            crime_name = "THEFT"
        elif "pocso" in query.lower():
            crime_name = "POCSO"
    
    has_officer = entities.get('officer_name') is not None
    has_year = entities.get('year') is not None
    officer_name = stats.get('officer', 'Unknown')
    if officer_name:
        officer_name = officer_name.split('(')[0].strip()

    response = ""
    if 'list' in query_lower or 'show' in query_lower or 'display' in query_lower:
        has_officer = entities.get('officer_name') is not None
        has_year = entities.get('year') is not None

        if has_officer and officer_name:
            if crime_name:
                response = f"Officer {officer_name} handled {num_cases} {crime_name} case(s)."
            else:
                response = f"Officer {officer_name} handled {num_cases} case(s)."
        # ✅ HANDLE YEAR RANGE
        if crime_name and 'year_range' in entities:
            start, end = entities['year_range'][0], entities['year_range'][1]
            
        elif crime_name and has_year:
            year = entities.get('year')
            all_crime_labels = ", ".join(entities['crime_type'][:3]) if entities.get('crime_type') else crime_name
            response = f"There are {num_cases} case(s) for [{all_crime_labels}] reported in {year}."
        elif crime_name:
            all_crime_labels = ", ".join(entities['crime_type'][:3]) if entities.get('crime_type') else crime_name
            response = f"There are {num_cases} case(s) for [{all_crime_labels}] in the records."
        else:
            response = f"There are {num_cases} case(s) matching your criteria."

        if 'years' in stats and len(stats['years']) > 0:
            year_breakdown = ", ".join([f"{year} ({count})" for year, count in sorted(stats['years'].items())])
            response += f" Year breakdown: {year_breakdown}."

        if not has_officer and 'IOName' in filtered_df.columns:
            top_officers = filtered_df['IOName'].value_counts().head(3)
            if len(top_officers) > 0:
                officer_summary = ", ".join([
                    f"{name.split('(')[0].strip()} ({count})"
                    for name, count in top_officers.items()
                ])
                

        if 'crime_heads' in stats and len(stats['crime_heads']) > 0:
            crime_details = ", ".join([
                f"{crime} ({count})"
                for crime, count in list(stats['crime_heads'].items())[:3]
            ])
            response += f" Case types: {crime_details}."

        if 'total_arrested' in stats:
            response += f" Total arrests: {stats['total_arrested']}."

        return response
    # ✅ HANDLE "WHO HANDLED" QUERIES (NEW FIX)
    if "handled" in query_lower and "who" in query_lower:
        if len(filtered_df) == 0:
            return "No records found for your query"

        officers = filtered_df['IOName'].dropna().unique()

        if len(officers) == 0:
            return "No officer information available"

        officer_list = [o.split('(')[0].strip() for o in officers[:5]]

        return f"{len(officers)} officer(s) handled these cases: {', '.join(officer_list)}"

    if 'how many' in query_lower:
        # ✅ PRIORITY: YEAR RANGE
        if crime_name and 'year_range' in entities:
            start, end = entities['year_range'][0], entities['year_range'][1]
            if entities.get('crime_type'):
                crime_list = ", ".join(list(set(entities['crime_type']))[:3])
                return f"There are {num_cases} case(s) for [{crime_list}] reported between {start} and {end}"
            else:
                return f"There are {num_cases} case(s) reported between {start} and {end}"
        if has_officer and officer_name and 'arrest' in query_lower:
            arrests = stats.get('total_arrested', 0)
            return f"Officer {officer_name} made {arrests} arrest(s) across all cases."
        elif has_officer and officer_name:
            crime_label = f" {crime_name}" if crime_name else ""
            return f"Officer {officer_name} handled {num_cases}{crime_label} case(s)."
        elif crime_name and has_year:
            year = entities.get('year')
            all_crime_labels = ", ".join(entities['crime_type'][:3]) if entities.get('crime_type') else crime_name
            if officer_name:
                return f"In {year}, Officer {officer_name} handled {num_cases} case(s)"
            else:
                return f"There are {num_cases} {crime_name} case(s) reported in {year}"
        elif crime_name:
            return f"There are {num_cases} {crime_name} case(s) in the records."
        else:
            return f"There are {num_cases} case(s) matching your criteria."

    if 'what about' in query_lower or ('in' in query_lower and entities.get('year')):
        year = entities.get('year', 'that year')

        if officer_name:
            response = f"In {year}, Officer {officer_name} handled {num_cases} case(s)"
        else:
            response = f"There are {num_cases} case(s) reported in {year}"

        if 'crime_types' in stats and len(stats['crime_types']) > 0:
            top_crimes = list(stats['crime_types'].keys())[:3]
            response += f", including {', '.join(top_crimes)}."
        else:
            response += "."

        return response

    if officer_name:
        response = f"Officer {officer_name} handled {num_cases} case(s)"
    else:
        response = f"There are {num_cases} case(s) found in the database"

    if 'districts' in stats:
        response += f" in {', '.join(stats['districts'])} district(s)"

    if 'years' in stats:
        years_list = sorted(stats['years'].keys())
        if len(years_list) > 1:
            response += f" from {years_list[0]} to {years_list[-1]}"

    response += "."

    if 'crime_types' in stats and len(stats['crime_types']) > 0:
        response += f" The cases include: "
        crime_summary = [f"{k} ({v})" for k, v in list(stats['crime_types'].items())[:3]]
        response += ", ".join(crime_summary) + "."

    if 'total_accused' in stats and 'total_arrested' in stats:
        response += f" Total accused: {stats['total_accused']}, arrests made: {stats['total_arrested']}."

    return response

def answer_query(query):

    filtered_df = df.copy()

    print(f"\nProcessing query: {query}")

    query_lower = query.lower().strip()   # ✅ FIRST define this

    explain_mode = False

    if "explain" in query_lower or "what is that case" in query_lower:
        explain_mode = True

    # Step 1: Greeting handling



    greetings = ["hello", "hi", "hey", "good morning", "good evening"]
    if query_lower in greetings:
        return "Hello! I am CrimeIntel assistant. Ask me about crime records.", {}, []
    
    # treat "what are" as list intent if asking about crimes
    if "what are" in query_lower and "crime" in query_lower:
        is_list_query = True

    # ✅ HANDLE "WHO IS OFFICER" QUERY
    if "who is" in query_lower:
        name = re.sub(r'who\s+is', '', query_lower).strip()

        # clean query name
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.lower()

        filtered_df = df[df['IOName_clean'].str.contains(name, na=False)]

        if len(filtered_df) == 0:
            return "No information found for this officer", {}, []

        officer = filtered_df['IOName'].iloc[0]
        district = filtered_df['District_Name'].iloc[0]

        total_cases = len(filtered_df)
        years = sorted(filtered_df['FIR_YEAR'].unique())

        return f"{officer} is a police officer in {district}. They handled {total_cases} case(s) from {years[0]} to {years[-1]}.", {}, filtered_df
    # Step 2: Check if query contains meaningful keywords
    keywords = [
        "case", "cases", "crime", "crimes",
        "officer", "police",
        "theft", "murder", "rape", "kidnap",
        "year", "district","happened",
        "record", "records",
        "list", "show", "display",
        "data", "details",
        "total", "overall", "all",
        "motor", "vehicle", "accident", "pocso"
    ]

    def is_semantic_query(query):

        query = query.lower().strip()

        # =====================================================
        # STEP 1 — STRUCTURED QUERIES
        # These MUST NOT go to semantic search
        # =====================================================

        structured_patterns = [
            "how many",
            "count",
            "total",
            "list",
            "show",
            "display",
            "give all",
            "all cases",
            "what are",
            "what is",
            "which cases",
            "cases in",
            "cases from",
            "recorded in",
            "reported in",
            "what happened in",     # ✅ ADD
            "what happened during",# ✅ ADD
            "cases during",        # ✅ ADD
            "cases reported during"
        ]

        if any(pattern in query for pattern in structured_patterns):
            return False

        # =====================================================
        # STEP 2 — EXPLICIT SIMILARITY / PATTERN QUESTIONS
        # =====================================================

        semantic_patterns = [
            "is there any case",
            "similar case",
            "cases like this",
            "cases like that",
            "something similar",
            "any incident",
            "related case",
            "similar incident",
            "same type",
            "have this happened",
            "anything similar",
            "pattern like this",
            "similar to this",
            "similar to that"
        ]

        if any(pattern in query for pattern in semantic_patterns):
            return True

        # =====================================================
        # STEP 3 — NATURAL INCIDENT DESCRIPTION
        # =====================================================

        incident_patterns = [
            "someone",
            "somebody",
            "two people",
            "one person",
            "a man",
            "a woman",
            "a child",
            "a group",

            "ran away",
            "was attacked",
            "was kidnapped",
            "was robbed",
            "was murdered",
            "was assaulted",

            "kidnapped",
            "robbed",
            "murdered",
            "assaulted",
            "stole",
            "stolen",
            "hacked",
            "crashed"
        ]

        return any(pattern in query for pattern in incident_patterns)
    # ✅ CLEAR CONTEXT FOR NEW INDEPENDENT QUERY
    if not any(word in query_lower for word in ["that", "those", "them", "his", "her"]):
        chat_history.context.pop('last_officer', None)
    if "crime names" in query_lower or "crime types" in query_lower:
        unique_crimes = sorted(df['CrimeGroup_Name'].dropna().unique())[:10]
        return f"Some recorded crime types are: {', '.join(unique_crimes)}", {}, []
    
    if is_semantic_query(query):

        similar_cases = semantic_case_search(query)

        if len(similar_cases) == 0:
            return "No similar cases found.", {}, []

        response = f"Found {len(similar_cases)} similar case records.\n\n"

        for i, (_, row) in enumerate(similar_cases.iterrows(), start=1):

            response += f"""
    Case {i}
    Case No:{row.get('Case_No')}
    District: {row['District_Name']}
    Crime Type: {row['CrimeGroup_Name']}
    Offence: {row['CrimeHead_Name']}
    Place: {row['Place of Offence']}
    Area: {row['Village_Area_Name']}
    Year: {row['FIR_YEAR']}
    Act Section: {row['ActSection']}
    Officer: {row['IOName']}

    """
        return response, {}, similar_cases
    
    if not any(word in query_lower for word in keywords):
        return "Please ask a crime-related query like 'how many theft cases in 2020'", {}, []
    entities = extract_query_entities(query, chat_history.context)


    # 🔥🔥🔥 CASE NUMBER DIRECT HANDLING (ADD HERE)
    if 'case_no' in entities:
        case_no = entities['case_no']

        print("🔍 Searching Case:", case_no)

        result = df[df['Case_No'] == case_no]

        if len(result) == 0:
            return f"No case found with Case No {case_no}", {}, []

        return detailed_case_view(result), {}, result

    

    # ✅ Detect year range like "2019 to 2020"
    year_range = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
    if len(year_range) >= 2:
        entities['year_range'] = year_range
    print(f"Extracted entities: {entities}")
    print(f"Context: {chat_history.context}")


    print("✅ After officer filter:", len(filtered_df))   # ✅ DEBUG

    if entities['officer_name'] or entities['district'] or entities['year'] or entities['crime_type'] or 'case_no' in entities:
        print(f"Using direct DataFrame query")
        filtered_df = df.copy()

        if entities['officer_name']:

            name = entities['officer_name'].lower().strip()

            # Normalize query officer name
            name = re.sub(r'[^\w\s]', '', name)
            name = re.sub(r'\s+', ' ', name).strip()

            print("🔍 Searching officer:", name)

            # Match the complete normalized name
            filtered_df = filtered_df[
                filtered_df['IOName_clean'].str.contains(
                    name,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

            print(
                "✅ After officer filter:",
                len(filtered_df)
            )

            print("✅ After officer filter:", len(filtered_df))

        # ✅ FIRST: CRIME FILTER
        if entities['crime_type']:
            mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)

            for crime in entities['crime_type']:
                keyword = crime.lower().strip()

                mask |= filtered_df['CrimeGroup_Name'].str.lower().str.contains(keyword, na=False)
                mask |= filtered_df['CrimeHead_Name'].str.lower().str.contains(keyword, na=False)

            filtered_df = filtered_df[mask]
            print("✅ After crime filter:", len(filtered_df))

        # ✅ SECOND: GENERIC DISTRICT FILTER
        if entities['district']:

            filtered_df = filtered_df[
                filtered_df['District_Name'].isin(
                    entities['district']
                )
            ]

            print(
                "✅ After district filter:",
                len(filtered_df)
            )

        # ✅ SECOND: OFFICER FILTER
        if entities['officer_name']:
            name = entities['officer_name'].lower().strip()
            name_parts = name.split()

            mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)

            for part in name_parts:
                mask &= filtered_df['IOName'].str.lower().str.contains(part, na=False)

            filtered_df = filtered_df[mask]
            print("✅ After officer filter:", len(filtered_df))

        if entities.get('year') and 'year_range' not in entities:
            try:
                year = int(entities['year'])   # ✅ convert to int
                filtered_df = filtered_df[
                    filtered_df['FIR_YEAR'] == year
                ]
            except:
                pass
            print(f"   After year filter: {len(filtered_df)} rows")
            print(f"   After year filter: {len(filtered_df)} rows")
        # ✅ Apply year range filter
        if 'year_range' in entities:
            start = int(entities['year_range'][0])
            end = int(entities['year_range'][1])

            filtered_df = filtered_df[
                (filtered_df['FIR_YEAR'] >= start) &
                (filtered_df['FIR_YEAR'] <= end)
            ]

            print(f"   After year range filter: {len(filtered_df)} rows")

        filtered_df = filtered_df.reset_index(drop=True)
        print(filtered_df[['CrimeGroup_Name','FIR_YEAR','IOName']].head())

    else:
        # ✅ handle general queries
        total_cases = len(df)
        top_crimes = df['CrimeHead_Name'].value_counts().head(5)

        crime_list = ", ".join([f"{c} ({n})" for c, n in top_crimes.items()])

        return f"There are {total_cases} total cases in the dataset. Top crimes include: {crime_list}", {}, []

    # ✅ HANDLE "WHO HANDLED" QUERY (FINAL FIX)
    if "who" in query_lower and "handled" in query_lower:
        if len(filtered_df) == 0:
            return "No records found for your query", {}, []

        officers = filtered_df['IOName'].dropna().unique()

        if len(officers) == 0:
            return "No officer information available", {}, []

        officer_list = [o.split('(')[0].strip() for o in officers[:5]]

        return f"{len(officers)} officer(s) handled these cases: {', '.join(officer_list)}", {}, filtered_df
    print("DEBUG DATA:")
    print(filtered_df[['CrimeGroup_Name','FIR_YEAR','IOName']].head(10))
    facts, stats = generate_facts_from_records(filtered_df, entities, query)
    if explain_mode:
        if len(filtered_df) == 0:
            return "No case found for explanation. Try with different filters.", {}, []

        return summarize_case(filtered_df) + "\n\n" + detailed_case_view(filtered_df), {}, []

        summary = summarize_case(filtered_df)
        detail = detailed_case_view(filtered_df)

        return summary + "\n\n" + detail, {}, []


    answer = generate_conversational_response(
        stats, query, entities, chat_history.context, filtered_df
    )
    print(f" Final result: {len(filtered_df)} matching records")

    # ✅ DETECT LIST INTENT
    list_keywords = ["list", "show", "display", "give all", "all cases"]

    is_list_query = any(word in query_lower for word in list_keywords)

    if is_list_query:
        return answer, stats, filtered_df  # return table
    else:
        return answer, stats, pd.DataFrame()  # summary only

    chat_history.add_interaction(query, facts, answer, filtered_df)
    return answer, stats, filtered_df #filtered_df(10)

def get_bert_response(query):
    query_lower = query.lower()
    answer, stats, records = answer_query(query)
    print(f"\n🤖 Assistant: {answer}")

    table_data = None

    show_table_keywords = ["show", "list", "display", "table"]

    if len(records) > 0 and any(word in query_lower for word in show_table_keywords):
        cols = ['Case_No', 'District_Name', 'IOName', 'CrimeGroup_Name', 'CrimeHead_Name', 'FIR_YEAR']
        display_cols = [c for c in cols if c in records.columns]
        print(f"\n📄 Matching Records:")
        print(records[display_cols].to_string(index=False))

        table_data = {
            "headers": display_cols,
            "rows": records[display_cols].fillna("").astype(str).values.tolist()
        }

    print(f"{'='*70}")
    return answer, table_data

def semantic_case_search(query, top_k=100):

        # STEP 1 — semantic embedding
        query_embedding = embed_model.encode(
            [f"query: {query}"],
            convert_to_numpy=True
        )

        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, top_k)

        matched_records = df.iloc[indices[0]].copy()

        matched_records['semantic_score'] = distances[0]

        # STEP 2 — similarity percentage
        similarity_scores = []

        for score in distances[0]:

            percentage = round(float(score) * 100, 2)

            if percentage > 100:
                percentage = 100.0

            if percentage < 0:
                percentage = 0.0

            similarity_scores.append(percentage)

        matched_records['Match_Percentage'] = similarity_scores

        # STEP 3 — query understanding
        query_lower = query.lower()

        crime_map = {
            "robbery": ["ROBBERY", "DACOITY", "THEFT"],
            "robbed": ["ROBBERY", "DACOITY", "THEFT"],
            "stolen": ["THEFT", "ROBBERY"],
            "stole": ["THEFT", "ROBBERY"],

            "murder": ["MURDER", "HOMICIDE"],

            "rape": ["RAPE", "SEXUAL"],

            "kidnap": [
                "KIDNAPPING",
                "ABDUCTION"
            ],

            "kidnapped": [
                "KIDNAPPING",
                "ABDUCTION"
            ],

            "child": [
                "POCSO",
                "KIDNAPPING",
                "ABDUCTION"
            ],

            "missing": [
                "MISSING PERSON",
                "KIDNAPPING"
            ],

            "assault": ["ASSAULT", "HURT"],
            "fight": ["ASSAULT", "HURT"],
            "gang": ["ASSAULT", "RIOTS"],

            "hack": ["CYBER", "CHEATING"],
            "hacked": ["CYBER", "CHEATING"],
            "fraud": ["CHEATING", "CYBER"],

            "drug": ["NDPS", "DRUG"],

            "accident": ["ACCIDENT"],

            "harassment": ["HARASSMENT", "SEXUAL"],

        

            "run away": [
                "MISSING PERSON",
                "KIDNAPPING"
            ],

            "missing": [
                "MISSING PERSON",
                "KIDNAPPING"
            ]
        }

        detected_crimes = []

        for word, aliases in crime_map.items():

            if word in query_lower:
                detected_crimes.extend(aliases)

        # STEP 4 — crime filtering
        if detected_crimes:

            mask = pd.Series(
                [False] * len(matched_records),
                index=matched_records.index
            )

            for crime in detected_crimes:

                mask |= matched_records['CrimeGroup_Name'] \
                    .str.contains(crime, case=False, na=False)

                mask |= matched_records['CrimeHead_Name'] \
                    .str.contains(crime, case=False, na=False)

                # ✅ NEW DESCRIPTION FILTER
                if 'Case_Description' in matched_records.columns:

                    mask |= matched_records['Case_Description'] \
                        .str.contains(crime, case=False, na=False)
                    
                mask |= matched_records['Village_Area_Name'] \
                    .str.contains(crime, case=False, na=False)
                mask |= matched_records['Place of Offence'] \
                    .str.contains(crime, case=False, na=False)

            filtered = matched_records[mask]

            if len(filtered) > 0:
                matched_records = filtered

        # STEP 5 — city filtering
        # STEP 5 — GENERIC DISTRICT FILTER
        districts = (
            df['District_Name']
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        matched_districts = []

        for district in districts:

            if district.lower() in query_lower:
                matched_districts.append(district)

        if matched_districts:

            district_filtered = matched_records[
                matched_records['District_Name'].isin(matched_districts)
            ]

            if len(district_filtered) > 0:
                matched_records = district_filtered

        # STEP 6 — ranking
        matched_records = matched_records.sort_values(
            by=[
                "Match_Percentage",
                "semantic_score"
            ],
            ascending=False
        )

        return matched_records.head(5)
# test_queries = [
#     "List the POCSO cases handled by officer Biradar"
# ]

# for query in test_queries:

#     print(f"\n{'='*70}")
#     print(f" User: {query}")
#     answer, stats, records = answer_query(query)
#     print(f"\nAssistant: {answer}")

#     if len(records) > 0 and len(records) <= 5:
#         print(f"\n Matching Records:")
#         print(records[['District_Name', 'IOName', 'CrimeGroup_Name', 'CrimeHead_Name', 'FIR_YEAR']].to_string(index=False))

#     print(f"{'='*70}")