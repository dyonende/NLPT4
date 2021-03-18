import re
import nltk
import sys
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer 



def list_of_features(filepath):
    lemmatizer = WordNetLemmatizer()
    feature_list = list()
    
    with open(filepath, 'r') as infile:
        content = infile.read()

    #filter out link
    content = re.sub('(?:(?:(?:ftp|http)[s]*:\/\/|www\.)[^\.]+\.[^ \n]+)','', content)
   
    #tokenize text
    tokens = word_tokenize(content)
    pos_tags = nltk.pos_tag(tokens)

    #determine features based on the tokens
    for token, pos_tag in zip(tokens, pos_tags):
        feature_dict = dict()
        feature_dict["token"] = token
        feature_dict["lemma"] = lemmatizer.lemmatize(token)
        feature_dict["POS-tag"] = str(pos_tag[1])
        feature_dict["coreference-information"] = ""
        feature_list.append(feature_dict)

    return feature_list


def list_dict_to_conll(list_of_dicts, filepath):
    """makes a conll file out of a list of dicts. Every dict is a row 
    
    param list_of_dicts: list of dicts
    param filepath: name of the filepath where the conll should be saved
    
    type list_of_dicts: list
    type filepath: string
    """
    
    with open(filepath, 'w') as outfile:
        header_row = 'Token\tLemma\tPOS-tag\tCoreference-information\n'
        outfile.write(header_row)
        for row in list_of_dicts:
            cell_values = row.values()
            line = "\t".join(cell_values) + '\n'
            outfile.write(line)
            

def main():
    
    args = sys.argv
    infile = args[1]
    outfile = args[2]
    
    #download required package
    nltk.download('averaged_perceptron_tagger')
    
    feature_list = list_of_features(infile)
    list_dict_to_conll(feature_list, outfile) 


if __name__ == '__main__':
    main()

