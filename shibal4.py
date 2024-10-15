# sample.py

import os
import ssl
import nltk
import json
import streamlit as st
import random
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
import string

# ---------------------------
# Setup and Downloads
# ---------------------------

# Setting up SSL context for nltk
ssl._create_default_https_context = ssl._create_unverified_context

# Downloading necessary NLTK data
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# ---------------------------
# Department Data
# ---------------------------

departments = {
    "cse": [
        "Dr. N. Bhalaji – Principal",
        "Dr. D. C. Joy Winnie Wise – Professor",
        "Dr. Lalitha R – Professor",
        "Dr. V. Anjana Devi – Professor",
        "Dr. R. Saravanan – Associate Professor",
        "Dr. Pandithurai O – Associate Professor",
        "Dr. T. Rajendran – Associate Professor",
        "Dr. Anwar Basha H. – Associate Professor",
        "Ashok M – Associate Professor",
        "C. Balaji – Associate Professor",
        "Dr. T. Nithya – Assistant Professor",
        "Dr. Ranjith Kumar M.V. – Assistant Professor",
        "C. Chairmakani – Assistant Professor",
        "AG. Noorul Julaiha – Assistant Professor",
        "B. Sriman – Assistant Professor",
        "R. Arunkumar – Assistant Professor",
        "Vijayalakshmi R. – Assistant Professor",
        "S. H. Annie Silviya – Assistant Professor",
        "S. Uma – Assistant Professor",
        "E. Venitha – Assistant Professor",
        "C.S. Somu – Assistant Professor",
        "G. Kavitha – Assistant Professor",
        "E. Pooja – Assistant Professor",
        "G. Sumathi – Assistant Professor",
        "R. Tamilselvan – Assistant Professor",
        "J. Praveen Kumar – Assistant Professor",
        "Dr. N. Indumathi – Assistant Professor",
        "Pugazhvendan I. – Professor of Practice",
        "Anantha Krishnan A. – Associate Professor of Practice",
        "Mugesh Hariharasudan – Assistant Professor of Practice"
    ],
    "ece": [
        "Dr. Sundar Rangarajan – Professor",
        "Dr. G. Nirmala Priya – Professor",
        "Dr. M. Malathi – Professor",
        "Dr. H. Sivaram – Associate Professor",
        "Dr. E. Sivanantham – Associate Professor",
        "Dr. M. Chitra – Associate Professor",
        "Dr. R. Sanmuga Sundaram – Associate Professor",
        "Dr. T. Roosefert Mohan – Associate Professor",
        "Chinnammal V – Assistant Professor",
        "Subashini V – Assistant Professor",
        "Kalyan Kumar G – Assistant Professor",
        "Kalaivani S – Assistant Professor",
        "Balaji A – Assistant Professor",
        "Malarvizhi C – Assistant Professor",
        "Vanathi A – Assistant Professor",
        "Charulatha Srinivasan – Assistant Professor",
        "Shofia Priyadharshini D. – Assistant Professor",
        "S. Sangeetha – Assistant Professor"
    ],
    "cce": [
        "Dr. C. Ganesh – Professor",
        "Dr. S. Ashok Kumar – Professor",
        "Dr. P. Sathish Kumar – Associate Professor",
        "Manimaran B. – Assistant Professor",
        "V. Sushmitha – Assistant Professor",
        "S. Bharath – Assistant Professor",
        "G. Saravanan – Assistant Professor",
        "N. Dharmaraj – Assistant Professor",
        "Vigneshvar D. – Associate Professor of Practice"
    ],
    "ee_vlsi_d&t": [
        "Dr. I. Chandra – Professor",
        "Franklin Telfer L – Assistant Professor",
        "Dr. Sheela S – Assistant Professor"
    ],
    "ec_vlsi": [
        "Dr. S. Manjula – Professor",
        "Jayamani K – Assistant Professor",
        "Monica M. – Assistant Professor",
        "Monikapreethi S.K. – Assistant Professor"
    ],
    "csbs": [
        "Dr. K. Ramkumar – Professor",
        "Dr. Subha S – Associate Professor",
        "Dr. S. Sridhar – Associate Professor",
        "M. Babu – Associate Professor",
        "Loganayaki D – Assistant Professor",
        "S. Sathiyan – Assistant Professor",
        "K. Fouzia Sulthana – Assistant Professor",
        "T. Pandiarajan – Assistant Professor",
        "R. Deepak – Assistant Professor",
        "M. Baskar – Assistant Professor",
        "K. Jayashree – Assistant Professor",
        "J. Lakshmikanth – Assistant Professor",
        "Dr. P. Kalaivani – Assistant Professor",
        "Dr. J. Maria Arockia Dass – Assistant Professor",
        "P. Jyothy – Professor of Practice"
    ],
    "ai_ds": [
        "Dr. N. Kanagavalli – Assistant Professor",
        "Dr. A. Arthi – Professor",
        "Dr. Srivenkateswaran C. – Professor",
        "Dr. M. Vivekanandan – Associate Professor",
        "Dr. B.N. Karthik – Associate Professor",
        "Dr. S. Niranjana – Assistant Professor",
        "R. Kennady – Assistant Professor",
        "S. Saranya – Assistant Professor",
        "S. Selvakumaran – Assistant Professor",
        "R. Saranya – Assistant Professor",
        "R. Kalaiyarasi – Assistant Professor",
        "V. Deepa – Assistant Professor",
        "B. Sasikala – Assistant Professor",
        "M. Bhavani – Assistant Professor",
        "S. Vaijayanthi – Assistant Professor",
        "S. Sahunthala – Assistant Professor",
        "T. Sam Paul – Assistant Professor",
        "G. Baby Saral – Assistant Professor",
        "M. Sneha – Assistant Professor",
        "Dr. Kalaiselvi S. – Assistant Professor",
        "A. Anbumani – Assistant Professor",
        "H. Hemal Babu – Assistant Professor",
        "V. Madhan – Assistant Professor",
        "Javis Jerald – Professor of Practice",
        "Nandhini – Assistant Professor of Practice",
        "K. Subashini – Assistant Professor of Practice",
        "Farzana B. – Assistant Professor of Practice"
    ],
    "cse_ai_ml": [
        "Dr. K. Regin Bose – Professor",
        "S. Shanthana – Assistant Professor",
        "F. Merlin Christo – Assistant Professor",
        "C. Gethara Gowri – Assistant Professor",
        "P. Somasundari – Assistant Professor",
        "K.G. Sara Rose – Professor of Practice"
    ],
    "mechanical": [
        "Dr. N. Pragadish – Professor",
        "Dr. Deepak Suresh – Professor",
        "Rajeswaran P. S. – Professor",
        "Dr. Rajesh Kanna S. K. – Professor",
        "Dr. M. Bakkiyaraj – Associate Professor",
        "Dr. Muthu G – Associate Professor",
        "Dr. Sai Krishnan G – Assistant Professor",
        "Dr. N. Sivashanmugam – Assistant Professor",
        "Dr. S. Bharani Kumar – Assistant Professor",
        "Srinivasan S. – Assistant Professor",
        "Vivek S – Assistant Professor"
    ]
}

# Create a reverse mapping: member to departments
member_to_departments = defaultdict(list)

for department, members in departments.items():
    for member in members:
        name = member.split("–")[0].strip().lower()
        member_to_departments[name].append(department)

# Mapping for first names
first_name_to_members = defaultdict(list)
for member_full_name in member_to_departments:
    first_name = member_full_name.split()[0]
    first_name_to_members[first_name].append(member_full_name)

# ---------------------------
# Intents Loading
# ---------------------------

def load_intents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents

intents_file_path = "D:\\Mridula\\stud chatbot\\stupid\\intents.json"  # Update with actual path
intents = load_intents(intents_file_path)

# ---------------------------
# Preprocessing and Model Training
# ---------------------------

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# ---------------------------
# Chatbot Functions
# ---------------------------

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extract_name(tokens):
    if 'does' in tokens:
        idx = tokens.index('does')
        name_tokens = tokens[idx + 1:]
        stopwords = {'belong', 'to', 'department'}
        name = ' '.join([token for token in name_tokens if token not in stopwords])
        return name.strip()
    elif 'belong' in tokens:
        idx = tokens.index('belong')
        name_tokens = tokens[idx + 1:]
        stopwords = {'to', 'department'}
        name = ' '.join([token for token in name_tokens if token not in stopwords])
        return name.strip()
    elif 'is' in tokens:
        idx = tokens.index('is')
        name_tokens = tokens[idx + 1:]
        stopwords = {'in', 'which', 'department'}
        name = ' '.join([token for token in name_tokens if token not in stopwords])
        return name.strip()
    else:
        return tokens[-1] if tokens else ""

def get_intent(user_input):
    tokens = word_tokenize(preprocess_text(user_input))
    
    if any(word in tokens for word in ["what", "list", "give", "show", "all"]):
        for department in departments.keys():
            department_tokens = department.replace("_", " ").split()
            if any(token in department_tokens for token in tokens):
                return ("list_members_intent", department)
    
    if any(word in tokens for word in ["which", "does", "belong", "belongs", "is"]):
        name = extract_name(tokens)
        if name:
            return ("find_department_intent", name)
    
    return ("unknown_intent", None)

def format_department_name(department_key):
    format_mapping = {
        "cse": "CSE",
        "ece": "ECE",
        "cce": "CCE (Computer and Communication)",
        "ee_vlsi_d&t": "EE VLSI D&T",
        "ec_vlsi": "EC VLSI",
        "csbs": "CSBS (Computer Science and Business Systems)",
        "ai_ds": "AI&DS (Artificial Intelligence and Data Science)",
        "cse_ai_ml": "CSE (AI&ML)",
        "mechanical": "Mechanical"
    }
    return format_mapping.get(department_key, department_key.upper())

def respond(intent, data):
    if intent == "list_members_intent":
        department = data
        department_display = format_department_name(department)
        members = departments.get(department, [])
        if members:
            response = f"The members of the {department_display} Department are:\n" + "\n".join(members)
        else:
            response = f"I don't have information about the {department_display} Department."
        return response
    
    elif intent == "find_department_intent":
        name_query = data.lower()
        matching_departments = set()
        matching_members = []

        for member_full_name, deps in member_to_departments.items():
            if name_query in member_full_name:
                matching_departments.update(deps)
                matching_members.append(member_full_name)
        
        if matching_departments:
            if len(matching_departments) == 1:
                department_display = format_department_name(next(iter(matching_departments)))
                response = f"The name '{data}' belongs to the {department_display} Department."
            else:
                departments_display = ", ".join([format_department_name(dep) for dep in sorted(matching_departments)])
                response = f"The name '{data}' belongs to the following departments: {departments_display}."
            return response
        else:
            return f"I couldn't find any departments for the name '{data}'. Please check the name and try again."
    
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

def chatbot_logic(input_text):
    input_text_lower = input_text.lower().strip()

    matched_tags = set()
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            regex = re.compile(r'\b' + re.escape(pattern.lower()) + r'\b')
            if regex.search(input_text_lower):
                matched_tags.add(intent['tag'])

    if matched_tags:
        responses = []
        for tag in matched_tags:
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    responses.append(random.choice(intent['responses']))
                    break
        combined_response = "\n\n".join(responses)
        return combined_response

    intent, data = get_intent(input_text)
    response = respond(intent, data)
    if intent != "unknown_intent":
        return response

    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

    return "I'm sorry, I didn't understand that. Can you please rephrase?"

# ---------------------------
# Streamlit UI
# ---------------------------

import streamlit as st

st.title("RIT Student Assistance Chatbot")

user_input = st.text_input("The RIT Student Assistance Chatbot is designed to help students with quick answers to common inquiries about academics, campus services, and more. It provides 24/7 support for tasks like finding information on courses, events, and campus resources, making student life easier and more efficient.")

if user_input:
    response = chatbot_logic(user_input)
    st.write(response)
