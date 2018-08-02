# Movie Analysis

This program takes wikidatas and calculates and plots correlation of different criterias to a successful film

the expected input files are required in the same directory as the program:
    wikidata-movies.json.gz
    rotten-tomatoes.json.gz
    omdb-data.json.gz
    genres.json.gz
  
the expected outputs are:
    ratings-correlation.png
    correlation coefficient of audience ratings and critics ratings
    
    ratings-award-correlation.png
    correlation coefficient of audience ratings and awards won
    
    score-profit-graph.png
    
required librarys are:
    pandas
    numpy
    scipy.stats
    math
    sys
    matplotlib.pyplot
    sklearn.model_selection
    
the program can be run as :

    ./movies_analysis.py