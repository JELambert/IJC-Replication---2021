#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: joshualambert
"""

import re
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import ldaseqmodel
import gensim
from gensim import corpora
import warnings
from fuzzywuzzy import fuzz

from visual import setting as se


se.set_rc_params()
warnings.simplefilter("ignore", DeprecationWarning)


def main():
    
    data = get_data()


    ############ VERY TIME CONSUMING!!! ####################
    ############ Get Fuzzy scores - to csv ####################
    df_combos(df)
    
    

    df = manage_data(data)
    
    scores = pd.read_csv('similarity scores.csv')
    scores = scores.loc[scores['score']>87]
    ids = list(scores.id2.unique())
    
    df['id'] = df.index +2
    df = df.loc[~df['id'].isin(ids)]
    
    plot_journal_counts(df)
    plot_yearly_counts(df)
    
    
    ########## DTM MODELS  #####################
    df = df.sort_values(by='Year',ascending=True)
    
    ## If you want to run lda model uncomment next line
    times = get_time_slice(df)
    
    doc_processed = df['abstract3'].map(preprocess)

    
    dictionary = corpora.Dictionary(doc_processed)
    #to prepapre a document term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_processed]
    
    
    ldaseq = ldaseqmodel.LdaSeqModel(corpus=doc_term_matrix, id2word = dictionary,
                                 time_slice=times, num_topics=25, chain_variance=0.05)

    #### Set the times
    times = ["1990-1992", "1993-1995", "1996-1998", "1999-2001", "2002-2004", "2005-2007", "2008-2010",
         "2011-2013", "2014-2016", "2017-2019"]


    ### Make the topics in a spreadsheet
    # full, twenty, twentyfive
    full = make_topics_time(times, ldaseq, number = "twentyfive")
    full['time period'] = full.apply(fix_times, axis=1)
    



def get_time_slice(df):
    
    """
    Sets the time slices for the lda seq
    """
    
    
    time_slice = [len(df.loc[(df.Year<=1992)]), len(df.loc[(df.Year>1992) & (df.Year<=1995)]), len(df.loc[(df.Year>1995) & (df.Year<=1998)]),
              len(df.loc[(df.Year>1998) & (df.Year<=2001)]), len(df.loc[(df.Year>2001) & (df.Year<=2004)]),
              len(df.loc[(df.Year>2004) & (df.Year<=2007)]), len(df.loc[(df.Year>2007) & (df.Year<=2010)]), 
              len(df.loc[(df.Year>2010) & (df.Year<=2013)]), len(df.loc[(df.Year>2013) & (df.Year<=2016)]),
              len(df.loc[(df.Year>2016)])]
    
    return time_slice

def preprocess(text):
    
    """
    Some pre processing by taking out stop words
    """
    
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(token)
    return result




def fix_times(x):
    
    """
    To better make graphs added these labels
    """
    
    
    if x['time'] == "1":
        return "1990-1992"
    elif x['time'] == "2":
        return "1993-1995"
    elif x['time'] == "3":
        return "1996-1998"
    elif x['time'] =="4":
        return "1999-2001"
    elif x['time'] =="5":
        return "2002-2004"
    elif x['time'] =="6":
        return "2005-2007"
    elif x['time'] == "7":
        return "2008-2010"
    elif x['time'] == "8":
        return  "2011-2013"
    elif x['time'] == "9":
        return "2014-2016"
    elif x['time']=="10":
        return "2018-2019"


def make_topics_time(times, ldaseq, number="full"):
    
    """
    Call the function for the particular topic number. 
    Returns a data frame with important information on topics
    by time.
    """
    
    full = []
    for i in range(len(times)):
        
        if number =="full":
            data = get_topics_by_time(ldaseq, i)
            full.append(data)
        elif number =="twenty":
            data = get_topics_by_time20(ldaseq, i)
            full.append(data)
        elif number == "twentyfive":
            data = get_topics_by_time25(ldaseq, i)
            full.append(data)
            
    return pd.concat(full)


def plot_yearly_counts(df):
    
    """
    The yearly plot with cumulative count.
    """
    
    
    
    df['count'] = 1
    yearly = df.groupby('Year')['count'].sum().reset_index()
    fig, ax= plt.subplots(figsize=(10,8))
    sns.set(style="darkgrid")
    sns.lineplot(x = 'Year', y = 'count', data=yearly)
    plt.savefig('basic plots/yearlycounts.png')
    
    
def plot_journal_counts(df):
    
    """
    The plot of journal counts with only the top fifteen journals.
    """
    
    
    df['count'] = 1
    journals = df.groupby('Source title')['count'].sum().reset_index()
    journals.sort_values(by='count', ascending=False, inplace=True)
    top40 = journals[:15]

    fig, ax= plt.subplots(figsize=(20, 12))
    ax = sns.barplot(x="count", y="Source title", data=top40)
    plt.yticks(fontsize=14)
    plt.xlabel("")
    plt.ylabel("")
    plt.subplots_adjust(left=0.4)
    plt.savefig('basic plots/journalcounts.png')



def manage_data(df):
    
    """
    A catch all to do some extra processing and also remove particular
    words that were not of use to the analysis.
    """
    
    
    df['Source title'] = df.apply(fix_source, axis=1)
    df['abstract1'] = df.apply(split_copy, axis=1)
    df['abstract2'] = df['abstract1'].map(lambda x: re.sub("[,\.!?-]", "", x))
    df['abstract2'] = df['abstract2'].map(lambda x: re.sub("\'", '', x))
    df['abstract2'] = df['abstract2'].map(lambda x: re.sub("\(", '', x))
    df['abstract2'] = df['abstract2'].map(lambda x: re.sub("\)", '', x))
    df['abstract2'] = df['abstract2'].map(lambda x: re.sub(r'\d+', '', x))
    df['abstract2'] = df['abstract2'].map(lambda x: x.strip())
    df['abstract2'] = df['abstract2'].map(lambda x: x.lower())



    remove_words = ['author', 'suggest', 'factor', 'one', 'two', 'three', 'four', 
                    'five', 'six', 'seven', 'eight', 'nine', 'ten', 'example', 
                    'journal', 'likely', 'first', 'second', 'third', 'article', 'among',
                    'show', 
                    'elsevier', 'research', 'thus', 'new', 'copyright', 'within',
                    'often', 'among', 
                    'significant', 'way', 'argue', 'argues', 'literature', 'finally',
                    'work', 
                    'works', 'argue', 'although', 'finds', 'finding', 'find', 'hypothesis',
                    'hypothesize', 'reviewer', 'http', 'research', 'section', 'article',
                    
                    'studi', 'studying', 'studies', 'study', 'analysis', 'analy', 
                    'analyzing', 'results', 'result', 'response', 'responses',
                    'respond', 'paper', 'papers', 'ie', 'eg', 'may', 'could', 'should', 
                    'would', 'let', 'must', 'else', 'hence', 'however', 'let', 'must', 
                    "nt", 'otherwise', 'shall', 'since', 'therefore', ',',  '-', 
                    'professor', 'hypothesis', 'hypothesize'
                    ]



    stop_words = list(stopwords.words('english'))
    
    pat = '|'.join([r'\b{}\b'.format(w) for w in remove_words])
    pat2 = '|'.join([r'\b{}\b'.format(w) for w in stop_words])
    
    df['abstract4'] = df['abstract2'].str.replace(pat, '')
    df['abstract4'] = df['abstract4'].str.replace(pat2, '')
    
    df['abstract5'] = df.apply(lemmatize_text, axis=1)


    df['abstract3'] = df['abstract5'].apply(lambda x: ' '.join([str(i) for i in x]))

    df = df[['Authors', 'Title', 'Year', 'Source title', 'DOI', 'Abstract',
       'Author Keywords', 'Document Type', 'Source', 'abstract3']]

    return df

def lemmatize_text(x):
    
    """
    Lemmatization function
    """
    
    lemmatizer=WordNetLemmatizer()

    input_str=word_tokenize(x['abstract4'])
    
    return [lemmatizer.lemmatize(w) for w in input_str]


def get_data():
        
    data = pd.read_csv('clean.csv')
    return data
    
    
def split_copy(x):
    
    try:
        return x['Abstract'].split("©")[0]
    except IndexError:
        return x['Abstract']
    


def fix_source(x):
    
    """
    Some sources were mismatched. This helps get them all unified.
    """
    
    if x['Source title'] == "ADMINISTRATION & SOCIETY":
        return "Administration and Society"
    
    elif x['Source title'] == "Agricultural Economics (United Kingdom)":
        return "Agricultural Economics"
    
    elif x['Source title'] == "AGRICULTURE ECOSYSTEMS & ENVIRONMENT":
        return "Agriculture, Ecosystems and Environment"

    elif x['Source title'] == "CANADIAN JOURNAL OF POLITICAL SCIENCE-REVUE CANADIENNE DE SCIENCE POLITIQUE":
        return "Canadian Journal of Political Science"    
    
    elif x['Source title'] == "CIRIEC-ESPANA REVISTA DE ECONOMIA PUBLICA SOCIAL Y COOPERATIVA":
        return "CIRIEC-Espana Revista de Economia Publica, Social y Cooperativa"
    
    elif x['Source title'] == "CITY & COMMUNITY":
        return "City and Community"

    elif x['Source title'] == "CONSERVATION & SOCIETY":
        return "Conservation and Society"
    
    elif x['Source title'] == "Conservation biology : the journal of the Society for Conservation Biology":
        return "Conservation Biology"

    elif x['Source title'] == "DESARROLLO ECONOMICO-REVISTA DE CIENCIAS SOCIALES":
        return "Desarrollo Economico"
    
    elif x['Source title'] == "Economic Development & Cultural Change":
        return "Economic Development and Cultural Change"
    
    elif x['Source title'] == "ECONOMICS & HUMAN BIOLOGY":
        return "Economics and Human Biology"
    
    elif x['Source title'] == "ENERGY RESEARCH & SOCIAL SCIENCE":
        return "Energy Research and Social Science"
    
    elif x['Source title'] == "ENVIRONMENT AND PLANNING A":
        return "Environment & Planning A"
    
    elif x['Source title'] == "ENVIRONMENT AND PLANNING A-ECONOMY AND SPACE":
        return "Environment & Planning A"
    
    elif x['Source title'] == "Environment & Planning C: Government & Policy":
        return "Environment and Planning C: Government and Policy"
    
    elif x['Source title'] == "ENVIRONMENT AND PLANNING C-GOVERNMENT AND POLICY":
        return "Environment and Planning C: Government and Policy"

    elif x['Source title'] == "ENVIRONMENT DEVELOPMENT AND SUSTAINABILITY":
        return "Environment, Development and Sustainability"
    
    elif x['Source title'] == "Environmental & Resource Economics":
        return "Environmental and Resource Economics"
        
    elif x['Source title'] == "ENVIRONMENTAL MODELLING & SOFTWARE":
        return "Environmental Modelling and Software"
        
    elif x['Source title'] == "ENVIRONMENTAL SCIENCE & POLICY":
        return "Environmental Science and Policy"
        
    elif x['Source title'] == "ETHICS POLICY & ENVIRONMENT":
        return "Ethics, Policy and Environment"
        
    elif x['Source title'] == "EURE-REVISTA LATINOAMERICANA DE ESTUDIOS URBANO REGIONALES":
        return "Eure"
        
    elif x['Source title'] == "EXEMPLARIA-A JOURNAL OF THEORY IN MEDIEVAL AND RENAISSANCE STUDIES":
        return "Exemplaria"
        
    elif x['Source title'] == "FOCAAL-JOURNAL OF GLOBAL AND HISTORICAL ANTHROPOLOGY":
        return "Focaal"
        
    elif x['Source title'] == "GAIA-ECOLOGICAL PERSPECTIVES FOR SCIENCE AND SOCIETY":
        return "GAIA"

    elif x['Source title'] == "GENDER PLACE AND CULTURE":
        return "Gender, Place and Culture"

    elif x['Source title'] == "Geografisk Tidsskrift - Danish Journal of Geography":
        return "Geografisk Tidsskrift"
        
    elif x['Source title'] == "GEOGRAFISK TIDSSKRIFT-DANISH JOURNAL OF GEOGRAPHY":
        return "Geografisk Tidsskrift"
        
    elif x['Source title'] == "GEOSCIENCES":
        return "Geosciences (Switzerland)"
        
    elif x['Source title'] == "GLOBAL ENVIRONMENTAL CHANGE-HUMAN AND POLICY DIMENSIONS":
        return "Global Environmental Change"
    
    elif x['Source title'] == "HISTORICAL SOCIAL RESEARCH-HISTORISCHE SOZIALFORSCHUNG":
        return "Historical Social Research"    
                
    elif x['Source title'] == "HUMAN NATURE-AN INTERDISCIPLINARY BIOSOCIAL PERSPECTIVE":
        return "Human Nature"                            
    
    elif x['Source title'] == "IDS BULLETIN-INSTITUTE OF DEVELOPMENT STUDIES":
        return "IDS Bulletin"    
    
    elif x['Source title'] == "INTERNATIONAL JOURNAL OF ENTREPRENEURIAL BEHAVIOR & RESEARCH":
        return "International Journal of Entrepreneurial Behaviour and Research"    
    
    elif x['Source title'] == "JASSS-THE JOURNAL OF ARTIFICIAL SOCIETIES AND SOCIAL SIMULATION":
        return "JASSS"
    
    elif x['Source title'] == "JOURNAL OF AGRICULTURAL & ENVIRONMENTAL ETHICS":
        return "Journal of Agricultural and Environmental Ethics"    
    
    elif x['Source title'] == "Journal of Agriculture and Rural Development in the Tropics and Subtropics, Supplement":
        return "JOURNAL OF AGRICULTURE AND RURAL DEVELOPMENT IN THE TROPICS AND SUBTROPICS"
    
    elif x['Source title'] == "JOURNAL OF ECONOMIC BEHAVIOR & ORGANIZATION":
        return "Journal of Economic Behavior and Organization"    
    
    elif x['Source title'] == "Journal of Enterprising Communities":
        return "Journal of Enterprising Communities: People and Places in the Global Economy"    

    elif x['Source title'] == "JOURNAL OF ENVIRONMENT & DEVELOPMENT":
        return "Journal of Environment and Development"    

    elif x['Source title'] == "JOURNAL OF INSTITUTIONAL AND THEORETICAL ECONOMICS-ZEITSCHRIFT FUR DIE GESAMTE STAATSWISSENSCHAFT":
        return "Journal of Institutional and Theoretical Economics"    
    
    elif x['Source title'] == "Land Degradation & Development":
        return "Land Degradation and Development"    
    
    elif x['Source title'] == "MOBISYS'18: PROCEEDINGS OF THE 16TH ACM INTERNATIONAL CONFERENCE ON MOBILE SYSTEMS, APPLICATIONS, AND SERVICES":
        return "MobiSys 2018 - Proceedings of the 16th ACM International Conference on Mobile Systems, Applications, and Services"    
    
    elif x['Source title'] == "NATURE & RESOURCES":
        return "Nature and Resources"    
    
    elif x['Source title'] == "NATURE + CULTURE":
        return "Nature and Culture"    
    
    elif x['Source title'] == "OCEAN & COASTAL MANAGEMENT":
        return "Ocean and Coastal Management"    
    
    elif x['Source title'] == "PASTORALISM-RESEARCH POLICY AND PRACTICE":
        return "Pastoralism"    
    
    elif x['Source title'] == "PHILOSOPHICAL TRANSACTIONS OF THE ROYAL SOCIETY B-BIOLOGICAL SCIENCES":
        return "Philosophical Transactions of the Royal Society B: Biological Sciences"    
    
    elif x['Source title'] == "Proceedings - 2011 5th IEEE Conference on Self-Adaptive and Self-Organizing Systems Workshops, SASOW 2011":
        return "Proceedings - 2011 5th IEEE International Conference on Self-Adaptive and Self-Organizing Systems, SASO 2011"    
    
    elif x['Source title'] == "Psychological Science in the Public Interest, Supplement":
        return "PSYCHOLOGICAL SCIENCE IN THE PUBLIC INTEREST"    
    
    elif x['Source title'] == "RANGELAND ECOLOGY & MANAGEMENT":
        return "Rangeland Ecology and Management"    
    
    elif x['Source title'] == "REVUE DE GEOGRAPHIE ALPINE-JOURNAL OF ALPINE RESEARCH":
        return "Revue de Geographie Alpine"    
    
    elif x['Source title'] == "Simulation & Gaming":
        return "Simulation and Gaming"        

    elif x['Source title'] == "SOCIAL MEDIA + SOCIETY":
        return "Social Media and Society"        

    elif x['Source title'] == "SOCIETY & NATURAL RESOURCES":
        return "Society and Natural Resources"        
    
    elif x['Source title'] == "SOUTH AFRICAN JOURNAL OF MARINE SCIENCE-SUID-AFRIKAANSE TYDSKRIF VIR SEEWETENSKAP":
        return "South African Journal of Marine Science"        
    
    elif x['Source title'] == "South American Camelids Research: Proceedings of the 4th European Symposium on South American Camelids and DECAMA European Seminar":
        return "South American Camelids Research Vol 1, Proceedings"        
    
    elif x['Source title'] == "SOZIALE WELT-ZEITSCHRIFT FUR SOZIALWISSENSCHAFTLICHE FORSCHUNG UND PRAXIS":
        return "Soziale Welt"        
    
    elif x['Source title'] == "Sustainability (Switzerland)":
        return "Sustainability"        
    
    elif x['Source title'] == "TECHNOLOGY ANALYSIS & STRATEGIC MANAGEMENT":
        return "Technology Analysis and Strategic Management"        
    
    elif x['Source title'] == "THEORY CULTURE & SOCIETY":
        return "Theory, Culture and Society"        
    
    elif x['Source title'] == "Theory, Culture & Society":
        return "Theory, Culture and Society"        
    
    elif x['Source title'] == "TRANSFORMING GOVERNMENT- PEOPLE PROCESS AND POLICY":
        return "Transforming Government: People, Process and Policy"  
    
    elif x['Source title'] == "Water (Switzerland)":
        return "WATER"  
    
    elif x['Source title'] == "WATER ALTERNATIVES-AN INTERDISCIPLINARY JOURNAL ON WATER POLITICS AND DEVELOPMENT":
        return "Water Alternatives"  
    
    elif x['Source title'] == "WORLDVIEWS-GLOBAL RELIGIONS CULTURE AND ECOLOGY":
        return "Worldviews: Environment, Culture, Religion"  
    
    elif x['Source title'] == "INTERNATIONAL JOURNAL OF THE COMMONS":
        return "International Journal of the Commons"      

    elif x['Source title'] == "ECOLOGY AND SOCIETY":
        return "Ecology and Society"    
    
    elif x['Source title'] == "HUMAN ECOLOGY":
        return "Human Ecology"    
 
    else:
        return x['Source title']




def get_topics_by_time25(ldaseq, time):
    
    """
    This function gets all the topic infomration from the 
    dynamic topic model.
    """
    
    t = time + 1
    
    timestr = str(t)
    timee = time
    
    topic1 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[0]:
        #print(i[0])
        topic1.append(("topic 1", i[0], i[1], timestr))
    topic1_df = pd.DataFrame(topic1, columns=['topic number', 'word', 'score', 'time'])    
    
    
    topic2 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[1]:
        #print(i[0])
        topic2.append(("topic 2", i[0], i[1], timestr))
    topic2_df = pd.DataFrame(topic2, columns=['topic number', 'word', 'score', 'time'])    
        
    topic3 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[2]:
        #print(i[0])
        topic3.append(("topic 3", i[0], i[1], timestr))
    topic3_df = pd.DataFrame(topic3, columns=['topic number', 'word', 'score', 'time'])    
        
    topic4 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[3]:
        #print(i[0])
        topic4.append(("topic 4", i[0], i[1], timestr))
    topic4_df = pd.DataFrame(topic4, columns=['topic number', 'word', 'score', 'time'])    
        
    topic5 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[4]:
        #print(i[0])
        topic5.append(("topic 5", i[0], i[1], timestr))
    topic5_df = pd.DataFrame(topic5, columns=['topic number', 'word', 'score', 'time'])    
    
    topic6 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[5]:
        #print(i[0])
        topic6.append(("topic 6", i[0], i[1], timestr))
    topic6_df = pd.DataFrame(topic6, columns=['topic number', 'word', 'score', 'time'])    
    
    topic7 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[6]:
        #print(i[0])
        topic7.append(("topic 7", i[0], i[1], timestr))
    topic7_df = pd.DataFrame(topic7, columns=['topic number', 'word', 'score', 'time'])    
    
    topic8 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[7]:
        #print(i[0])
        topic8.append(("topic 8", i[0], i[1], timestr))
    topic8_df = pd.DataFrame(topic8, columns=['topic number', 'word', 'score', 'time'])    
    
    topic9 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[8]:
        #print(i[0])
        topic9.append(("topic 9", i[0], i[1], timestr))
    topic9_df = pd.DataFrame(topic9, columns=['topic number', 'word', 'score', 'time'])    
        
    topic10 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[9]:
        #print(i[0])
        topic10.append(("topic 10", i[0], i[1], timestr))
    topic10_df = pd.DataFrame(topic10, columns=['topic number', 'word', 'score', 'time'])    
        

    topic11 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[10]:
        #print(i[0])
        topic11.append(("topic 11", i[0], i[1], timestr))
    topic11_df = pd.DataFrame(topic11, columns=['topic number', 'word', 'score', 'time'])    
    
    
    topic12 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[11]:
        #print(i[0])
        topic12.append(("topic 12", i[0], i[1], timestr))
    topic12_df = pd.DataFrame(topic12, columns=['topic number', 'word', 'score', 'time'])    
        
    topic13 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[12]:
        #print(i[0])
        topic13.append(("topic 13", i[0], i[1], timestr))
    topic13_df = pd.DataFrame(topic13, columns=['topic number', 'word', 'score', 'time'])    
        
    topic14 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[13]:
        #print(i[0])
        topic14.append(("topic 14", i[0], i[1], timestr))
    topic14_df = pd.DataFrame(topic14, columns=['topic number', 'word', 'score', 'time'])    
        
    topic15 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[14]:
        #print(i[0])
        topic15.append(("topic 15", i[0], i[1], timestr))
    topic15_df = pd.DataFrame(topic15, columns=['topic number', 'word', 'score', 'time'])    
    
    topic16 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[15]:
        #print(i[0])
        topic16.append(("topic 16", i[0], i[1], timestr))
    topic16_df = pd.DataFrame(topic16, columns=['topic number', 'word', 'score', 'time'])    
    
    topic17 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[16]:
        #print(i[0])
        topic17.append(("topic 17", i[0], i[1], timestr))
    topic17_df = pd.DataFrame(topic17, columns=['topic number', 'word', 'score', 'time'])    
    
    topic18 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[17]:
        #print(i[0])
        topic18.append(("topic 18", i[0], i[1], timestr))
    topic18_df = pd.DataFrame(topic18, columns=['topic number', 'word', 'score', 'time'])    
    
    topic19 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[18]:
        #print(i[0])
        topic19.append(("topic 19", i[0], i[1], timestr))
    topic19_df = pd.DataFrame(topic19, columns=['topic number', 'word', 'score', 'time'])    
        
    topic20 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[19]:
        #print(i[0])
        topic20.append(("topic 20", i[0], i[1], timestr))
    topic20_df = pd.DataFrame(topic20, columns=['topic number', 'word', 'score', 'time'])    



    topic21 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[20]:
        #print(i[0])
        topic21.append(("topic 21", i[0], i[1], timestr))
    topic21_df = pd.DataFrame(topic21, columns=['topic number', 'word', 'score', 'time'])    
    
    
    topic22 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[21]:
        #print(i[0])
        topic22.append(("topic 22", i[0], i[1], timestr))
    topic22_df = pd.DataFrame(topic22, columns=['topic number', 'word', 'score', 'time'])    
        
    topic23 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[22]:
        #print(i[0])
        topic23.append(("topic 23", i[0], i[1], timestr))
    topic23_df = pd.DataFrame(topic23, columns=['topic number', 'word', 'score', 'time'])    
        
    topic24 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[23]:
        #print(i[0])
        topic24.append(("topic 24", i[0], i[1], timestr))
    topic24_df = pd.DataFrame(topic24, columns=['topic number', 'word', 'score', 'time'])    
        
    topic25 = []
    for i in ldaseq.print_topics(time=timee, top_terms=30)[24]:
        #print(i[0])
        topic25.append(("topic 25", i[0], i[1], timestr))
    topic25_df = pd.DataFrame(topic25, columns=['topic number', 'word', 'score', 'time'])         
    
    data2 = pd.concat([topic1_df, topic2_df, topic3_df, topic4_df, topic5_df,
                       topic6_df, topic7_df, topic8_df, topic9_df, topic10_df,
                       topic11_df, topic12_df, topic13_df, topic14_df, topic15_df,
                       topic16_df, topic17_df, topic18_df, topic19_df, topic20_df,
                       topic21_df, topic22_df, topic23_df, topic24_df, topic25_df,
                       ])
    

    return data2


def combos(df):
    
    return list(itertools.combinations(df['id'], 2))


def df_combos(df):
    
    """
    get similarity scores
    """
    
    df['id'] = df.index +2
    all_combos = combos(df)
    
    fuzz_scores = []
    
    for i in all_combos:
        
        score = fuzz.ratio(df.loc[df.id==i[0]]['Title'].values, df.loc[df.id==i[1]]['Title'].values)
        
        id1 = i[0]
        id2 = i[1]
        
        fuzz_scores.append((score, id1, id2))
    
    
    score_df = pd.DataFrame(fuzz_scores, columns=['score', 'id1', 'id2'])
    score_df.sort_values(by='score', ascending=False, inplace =True)
    score_min = score_df[:1000]
    score_min.to_csv('similarity scores.csv')       
    