# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import re
import numpy as np
from numpy import linalg as LA
import sys
from PorterStemmer import PorterStemmer
import itertools
import random
import string

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'moviebot'
        
        self.creative = creative
        # state variables in process()
        self.time_to_recommend = 0 # the system is ready to recommend
        self.user_wants_recommend = 0 # the user says 'yes'
        self.recommended_movies = []
        self.next_movie_to_recommend = 0
        self.just_prompted_spellcheck = 0
        self.spell_corrected_movie = ""
        self.line_before_spellcheck = ""
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.movie_titles = [i[0] for i in self.titles] # extract just the titles into a single array
        self.sentiment = movielens.sentiment()
        self.num_ratings = 0
        self.threshold = 0.5
        self.response_directory = {
          "zero_movies_starter": [
            "Oops! I can't tell if you forgot to put your movie title in quotation marks or didn't mention a movie at all. Try again please.",
            "Sorry, could you try again with the movie title in quotations?",
            "I'm scratching my head over here because I can't figure out what movie you're referring to. Try again, man!"
          ], 

          "zero_movies_creative": [
            "Oops! I couldn't find any movie in that statement. Try again please.",
            "Sorry, I wasn't able to understand the movie name from that sentence, could you try again?",
            "I couldn't find the movie you're talking about. It's not you, it's me! Try again though.",
            "I'm trying as hard as I can but still can't find that movie! Can you try again?"
          ], 

          "multiple_movies_starter": [
            "You've mentioned more than one movie. Can you please tell me about them one at a time?",
            "I'm sorry, I can't really handle too much external stimulus. :p Can you please try one movie at a time?",
            "There might be more than one movie with that name. Can you be more specific?",
          ],

          "closest_movie": [
            "D'oh. I couldn't find '{old}' in my records, did you mean '{new}'? (Yes or no)",
           "'{old}' doesn't exist in my database, did you mean '{new}'? (Yes or no)",
           "Hmm, I can't seem to find '{old}'. Did you mean '{new}'? (Yes or no)",
           ],

           "no_match": [
              "Unfortunately I wasn't able to find '{movie}' :(. Can you tell me your thoughts about another movie?",
              "Sorry man, I couldn't find '{movie}' in my database :(. Do you have another in mind?",
              "I can't find '{movie}', but maybe it wasn't meant to be. Let's try another movie."
           ],

           "spellcheck_fail": [
             "Oh okay, my bad. I don't think your movie exists in my database then. Could you try talking about another movie?",
             "Oh no, I must not have record of your movie then. Try talking about another movie."
           ],

           "liked_movie": [
             "You liked '{movie}', huh? Me too! ",
             "I liked '{movie}' too! ",
             "I know! '{movie}' was good! ",
             "I enjoyed '{movie}' too! ",
             "Wow, I thought I was the only one that enjoyed '{movie}'! ",
             "Cool, so you liked '{movie}'. ",
             "I can't believe you liked '{movie}' too! "
           ],

            "really_liked_movie": [
             "Boy, you really liked '{movie}', huh? Me too! ",
             "I enjoyed '{movie}' a lot too! ",
             "I know! '{movie}' was great! ",
             "I know! I've watched '{movie}' at least 10 times! ",
             "Woot! '{movie}' is my favorite! ",
             "'{movie}' was absolutely bomb! ",
           ],

           "disliked_movie": [
             "Oh no, sounds like you didn't enjoy '{movie}' much, huh? ",
             "Guess you didn't like '{movie}' much. ",
            "Tell me about it :(. '{movie}' was a bummer. ",
            "I agree, I didn't like '{movie}' either. ",
             "Okay, you didn't like '{movie}'. ",
             "Seems like you didn'y enjoy '{movie}'. "
           ],

           "really_disliked_movie": [
             "Oh no, sounds like you didn't enjoy '{movie}' at all, huh? ",
             "Damn, you didn't like '{movie}' at all. ",
             "Really? You didn't like {movie}? Hmm. I didn't like it much either.  ",
             "Tell me about it :(. '{movie}' was such a big bummer. ",
             "Guess you didn't like '{movie}' at all. ",
           ],

           "couldnt_understand_yes_no": [
             "I don't think I understand, could you try typing 'yes' or 'no' again?",
             "Sorry, was that a 'yes' or 'no'?",
             "Errrr... what? 'yes' or 'no'?"
           ],

            "tell_me_more": [
             "Tell me about another movie.",
             "Okay, tell me about another one .. ",
             "What about more movies? Can you tell me about another one?"
           ],

           "arbitrary_inputs": [
             "I'm really not sure how to answer that. Can you try telling me about a movie? Try typing - I liked \"Limitless\"",
             "I don't know how to answer you. Try telling me about a movie with something like - I liked \"Limitless\"",
             "Uhh ... I am not sure what you mean, I'm just a poor Movie Bot. Try telling me about a movie with something like - I liked \"Limitless\"",
             "You got me! I don't really know. Try telling me about a movie with something like - I liked \"Limitless\""
           ],

           "greeting_creative": [
             "Hello! Tell me about a movie you've seen and whether you liked it or not. ",
              "Hey there. So, what movies did you see recently that you liked or disliked? ",
              "Howdy! So, tell me about your recent movie watches and if you liked them or not. Oh, and one by one please! "
           ],

           "greeting_starter": [
             "Since I'm a 'Starter Mode' bot, I can only understand if you put the name of the movie within double quotes",
             "Please remember to put the name of the movie within double quotes!",
             "Oh and please put the exact name of the movie in double quotes."
           ]

      

        }
        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################
        # Binarize the movie ratings before storing the binarized matrix.
        ratings = self.binarize(ratings)
        self.ratings = ratings
        self.user_ratings = np.zeros(self.ratings.shape[0])
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = random.choice(self.response_directory["greeting_creative"])
        if not self.creative: greeting_message = greeting_message + random.choice(self.response_directory["greeting_starter"])
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "It was nice to hear from you! Hope to chat again soon."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message
    
    def handle_recommendation(self, line):
        
        if self.time_to_recommend == 1:     
            if not line.lower() == "yes":
                return "Sorry, I don't understand. Please say yes to continue or :quit to exit."

            if self.user_wants_recommend == 0: # beginning of recommendations
                self.user_wants_recommend = 1
                print("Give me one moment while I figure out some movies to recommend...")
                self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
            if len(self.recommended_movies) == 0:
                self.time_to_recommend = 0
                return "Unfortunately I can't find any movies to recommend just yet. But let's keep going. Tell me about another movie."
            elif self.next_movie_to_recommend < len(self.recommended_movies):
                response = "I recommend \"" + self.movie_titles[self.recommended_movies[self.next_movie_to_recommend]] + "\"."
                self.next_movie_to_recommend = self.next_movie_to_recommend + 1
                
                if self.next_movie_to_recommend >= len(self.recommended_movies):
                    response = response + " Well that's all I have for now! Type :quit to exit."
                    self.time_to_recommend = 0
                    self.user_wants_recommend = 0
                else:
                    response = response + " Would you like another recommendation? (yes or :quit to exit)"
                return response
            elif self.next_movie_to_recommend >= len(self.recommended_movies):
                    response = response + " Well that's all I have for now! Type :quit to exit."
                    self.time_to_recommend = 0
                    self.user_wants_recommend = 0


        else:
            self.time_to_recommend = 1
            return "You've told me your opinions about 5 movies--awesome! Would you like me to recommend some movies? Type yes to continue."
    
    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def movie_not_found(self, movie, line):
          closest_movie = self.find_movies_closest_to_title(movie)
          if(len(closest_movie) == 0): # no close match found
                return random.choice(self.response_directory["no_match"]).format(movie=movie)
          else: # close match found, suggest any one
                self.line_before_spellcheck = line  # storing this to pass it to sentiment analysis in the next round
                new_movie = self.movie_titles[closest_movie[0]]
                self.spell_corrected_movie = new_movie
                self.just_prompted_spellcheck = 1
                return random.choice(self.response_directory["closest_movie"]).format(old=movie, new=new_movie)

    def handle_sentiment(self, movie, line, sentiment):
          if sentiment == 1:
              return random.choice(self.response_directory["liked_movie"]).format(movie=movie)
          elif sentiment == 2:
              return random.choice(self.response_directory["really_liked_movie"]).format(movie=movie)
          elif sentiment == -1:
              return random.choice(self.response_directory["disliked_movie"]).format(movie=movie)
          elif sentiment == -2:
              return random.choice(self.response_directory["really_disliked_movie"]).format(movie=movie)

    def process_starter(self, line):
        extracted_movies = self.extract_titles(line)

        if not self.creative:
            if self.time_to_recommend == 1:
                return self.handle_recommendation(line)
           
            if len(extracted_movies) == 0:
                return random.choice(self.response_directory["zero_movies_starter"])
            elif len(extracted_movies) > 1:
                return random.choice(self.response_directory["multiple_movies_starter"])

            movie = extracted_movies[0]
            movie_indices = self.find_movies_by_title(movie)

            if len(movie_indices) > 1:
                return "I noticed there are multiple movies called " + movie + ". Can you please add the year of the one you're talking about?"
            elif len(movie_indices) == 0:
                return "Unfortunately I wasn't able to find " + movie + ". :( Can you tell me your thoughts about another movie?"

            sentiment = self.extract_sentiment(line)
            if sentiment == 0:
                return "I can't tell if you liked " + movie + ". Can you tell me more of your thoughts on it?"
            else:
                response = self.handle_sentiment(movie, line, sentiment)
                if self.user_ratings[movie_indices[0]] == 0:
                    self.num_ratings = self.num_ratings + 1
                    self.user_ratings[movie_indices[0]] = sentiment
                else:
                    return "You've already talked about " + movie + ", so tell me about another one!"

                if self.num_ratings >= 5:
                    return self.handle_recommendation(line)
                else:
                    return response + random.choice(self.response_directory["tell_me_more"])

    def handle_arb_inputs(self, line):
        arbitrary_input_rules_1 = [not self.time_to_recommend == 1, not self.just_prompted_spellcheck == 1]
        arbitrary_input_rules_2 = [
        line.lower().startswith("what"), 
        line.lower().startswith("can"),
        line.lower().startswith("could"),
        line.lower().startswith("when"),
        line.lower().startswith("how")
        ]
        if any(arbitrary_input_rules_2) and all(arbitrary_input_rules_1): 
          return random.choice(self.response_directory["arbitrary_inputs"])
        else:
          return ""

    def process_creative(self, line):
            
            extracted_movies = self.extract_titles(line)
            input_for_sentiment = line
            # STEP 1: Check if its time to recommend
            if self.time_to_recommend == 1:
                return self.handle_recommendation(line)

            # STEP 2: Check if its time to get their approval on closest-spelled-movie suggestion                
            if self.just_prompted_spellcheck == 1:
                  if line.lower() == "yes": 
                      # SPELL SUGGESTION WORKED
                      extracted_movies = [self.spell_corrected_movie]      
                      self.spell_corrected_movie = "" 
                      self.just_prompted_spellcheck = 0
                      input_for_sentiment = self.line_before_spellcheck
                      
                  elif line.lower() == "no": 
                      # SPELL SUGGESTION DIDNT WORK
                      self.just_prompted_spellcheck = 0
                      self.spell_corrected_movie = ""
                      return random.choice(self.response_directory["spellcheck_fail"]) 

                  else: # their response wasn't "yes" or "no"
                      return random.choice(self.response_directory["couldnt_understand_yes_no"]) 
          
            # STEP 3: Handle no movies found, or too many movies found
            if len(extracted_movies) == 0:
                  arb = self.handle_arb_inputs(line) 
                  if arb == "": return random.choice(self.response_directory["zero_movies_creative"])
                  else: return arb
            elif len(extracted_movies) > 1:
                return random.choice(self.response_directory["multiple_movies_starter"])

            # STEP 4: Edge cases passed, get the movie from the database
            movie = extracted_movies[0]
            movie_indices = self.find_movies_by_title(movie)

            if len(movie_indices) == 0: 
                arb = self.handle_arb_inputs(line) 
                if arb == "": return self.movie_not_found(movie, line)
                else: return arb
            elif len(movie_indices) > 1: return "I noticed there are multiple movies called \"" + movie + "\". Can you please add the year of the one you're talking about?"

            # STEP 5: If movie found in database, record its sentiment
            else: sentiment = self.extract_sentiment(input_for_sentiment)
            
            if sentiment == 0: return "I can't tell if you liked " + movie + ". Can you tell me more of your thoughts on it?"
            else: 
                response = self.handle_sentiment(movie, input_for_sentiment, sentiment)
                if self.user_ratings[movie_indices[0]] == 0:
                    self.num_ratings = self.num_ratings + 1
                    self.user_ratings[movie_indices[0]] = sentiment
                else:
                    return "You've already talked about " + movie + ", so tell me about another one!"
           # STEP 6: If 5 movies registered, move on to making the recommendation
            if self.num_ratings >= 5:
                  return self.handle_recommendation(line)
            else:
                  return response + random.choice(self.response_directory["tell_me_more"])

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        if line.lower() == "who are you?": return "Well..."

        try:
          if self.creative: return self.process_creative(line) 
          else: return self.process_starter(line)
        except: 
          response = "I'm having a hard time understanding. Please tell me about a movie and whether you liked it or didn't."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        movies = []

        regex_with_quotes = r'\"(.*?)\"'
        movies = re.findall(regex_with_quotes, preprocessed_input)

        if self.creative:
          words = preprocessed_input.lower()
          words = words.split()
          # get combinations of words
          for i in range(len(words)+1):
            titles = list(itertools.combinations(words, i))
            for j in range(len(titles)):
              title = ""
              for word in titles[j]:
                word = word[0].upper() + word[1:]
                title += word + " "
              title = title[:len(title) - 1]
              if i >= 1:
                while title[len(title) - 1] in string.punctuation:
                  title = title[:len(title) - 1]
              result = self.find_helper(title)
              if result and result not in movies:
                movies.append(title)
        return movies
    
    def find_helper(self, title):
        movie_list = []
        title_parts = re.match("(?P<article>(the\s|an\s|a\s)?)(?P<movie>.*(?<!\(\d{4}\)))(?P<year>\(\d{4}\))?$", title, flags=re.IGNORECASE)
        if title_parts == None:
            return []
        article = title_parts.group('article').strip() if title_parts.group('article') != None else ""
        movie = title_parts.group('movie').strip() if title_parts.group('movie') else ""
        year = title_parts.group('year').strip() if title_parts.group('year') else ""
        patterns = []
        if year == "":
            patterns.append(re.compile((article + " " if article != "" else "") + re.escape(movie) + " \(\d{4}\)", flags=re.IGNORECASE))
            if article != "":
                patterns.append(re.compile(re.escape(movie) + ",[ ]?" + article + " \(\d{4}\)", flags=re.IGNORECASE))
        else:
            patterns.append(re.compile((article + " " if article != "" else "") + re.escape(movie) + " " + re.escape(year)))
            if article != "":
                patterns.append(re.compile(re.escape(movie) + ",[ ]?" + article + " " + re.escape(year), flags=re.IGNORECASE))
        for r in patterns:
            result = list(filter(r.match, self.movie_titles))
            if len(result) > 0:
                for movie in result:
                    movie_list.append(self.movie_titles.index(movie))
        return list(set(movie_list))
    
    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 1953]
        
        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        movie_list = []
        title_parts = re.match("(?P<article>(the\s|an\s|a\s)?)(?P<movie>.*(?<!\(\d{4}\)))(?P<year>\(\d{4}\))?$", title, flags=re.IGNORECASE)
        if title_parts == None:
            return []
        article = title_parts.group('article').strip() if title_parts.group('article') != None else ""
        movie = title_parts.group('movie').strip() if title_parts.group('movie') else ""
        year = title_parts.group('year').strip() if title_parts.group('year') else ""
        patterns = []
        patterns_creative = []
        patterns_disambiguate = []
        if year == "":
            patterns.append(re.compile((article + " " if article != "" else "") + re.escape(movie) + " \(\d{4}\)", flags=re.IGNORECASE))
            if article != "":
                patterns.append(re.compile(re.escape(movie) + ",[ ]?" + article + " \(\d{4}\)", flags=re.IGNORECASE))
        else:
            patterns.append(re.compile((article + " " if article != "" else "") + re.escape(movie) + " " + re.escape(year)))
            if article != "":
                patterns.append(re.compile(re.escape(movie) + ",[ ]?" + article + " " + re.escape(year), flags=re.IGNORECASE))
        if self.creative:
            foreign_alt_parts = re.match("(?P<article>(le |les |la |el |il |las |i |une |der |die |lo |los |das |de |un |en |den |det )?)(?P<movie>.*(?<!\(\d{4}\)))(?P<year>\(\d{4}\))?$", title, flags=re.IGNORECASE)
            article = foreign_alt_parts.group('article').strip() if foreign_alt_parts.group('article') != None else ""
            movie = foreign_alt_parts.group('movie').strip() if foreign_alt_parts.group('movie') else ""
            year = title_parts.group('year').strip() if title_parts.group('year') else ""
            #match = re.search("\([^\(\)]+\)",movie)
            if foreign_alt_parts != None:
                if year == "":
                    if article != "":
                        patterns_creative.append(re.compile("[^\(]+ \((?:a.k.a. )?" + article + " " + re.escape(movie) + "\) (?:\(\d{4}\))?", flags=re.IGNORECASE))

                        patterns_creative.append(re.compile("[^\(]+ \((?:a.k.a. )?" + re.escape(movie) + ", " + article + "\) (?:\(\d{4}\))?", flags=re.IGNORECASE))
                    else:
                        patterns_creative.append(re.compile("[^\(]+ \((?:a.k.a. )?" + re.escape(movie) + "\) (?:\(\d{4}\))?", flags=re.IGNORECASE))
                else:
                    if article != "":
                        patterns_creative.append(re.compile("[^\(]+ \((?:a.k.a. )?" + article + " " + re.escape(movie) + "\) " + re.escape(year), flags=re.IGNORECASE))

                        patterns_creative.append(re.compile("[^\(]+ \((?:a.k.a. )?" + re.escape(movie) + ", " + article + "\) " + re.escape(year), flags=re.IGNORECASE))
                    else:
                        patterns_creative.append(re.compile("[^\(]+ \((?:a.k.a. )?" + re.escape(movie) + "\) " + re.escape(year), flags=re.IGNORECASE))
            if year == "": 
                patterns_disambiguate.append(re.compile("^" + movie + "[^\w]", re.IGNORECASE)) # disambiguation part 1
        for r in patterns:
            result = list(filter(r.match, self.movie_titles))
            if len(result) > 0:
                for movie in result:
                    movie_list.append(self.movie_titles.index(movie))
        for r in patterns_disambiguate:
            result = list(filter(r.match, self.movie_titles))
            if len(result) > 0:
                for movie in result:
                    movie_list.append(self.movie_titles.index(movie))
        for r in patterns_creative:
            result = list(filter(r.match, self.movie_titles))
            if len(result) > 0:
                for movie in result:
                    match = re.match("(?P<article>(le |les |la |el |il |las |i |une |der |die |lo |los |das |de |un |en |den |det )?)(?P<movie>[^\(\)]+ \([^\(\)]+\) (?<!\(\d{4}\)))(?P<year>\(\d{4}\))?$", movie, flags=re.IGNORECASE)
                    if self.creative and foreign_alt_parts != None and match == None:
                        continue
                    movie_list.append(self.movie_titles.index(movie))

        movie_list = list(set(movie_list))
        movie_list.sort()
        return movie_list

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Supported lexicons so far: not, never, no, neither
        Supported contrapositives so far: yet, still, but

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        sentiments = dict()
        p = PorterStemmer()
        for key in self.sentiment:
            stemmed_key = p.stem(key, 0, len(key)-1)
            if self.sentiment[key] == 'pos':
                sentiments[stemmed_key] = 1
            elif self.sentiment[key] == 'poss':
                sentiments[stemmed_key] = 2
            elif self.sentiment[key] == 'neg':
                sentiments[stemmed_key] = -1
            elif self.sentiment[key] == 'negs':
                sentiments[stemmed_key] = -2

        # Fine-grained sentiment extraction
        strong_coeff = 2

        # Detect emotion using punctuation
        if preprocessed_input.count('!') + preprocessed_input.count('?') >= 2:
            strong_coeff = 2

        # For lines that do not have punctuations in the end
        if preprocessed_input and preprocessed_input[-1].isalpha():
          preprocessed_input = preprocessed_input + '\n'

        # Use Porter Stemmer on the input
        words = ''
        word = ''
        for c in preprocessed_input:
            if c.isalpha():
                word += c.lower()
            else:
                if word:
                    words += p.stem(word, 0, len(word)-1)
                    word = ''
                words += c.lower()

        # Remove the movie titles and puncuation
        words = re.sub('"(.*?)"', '', words)
        words = re.sub("[^a-zA-Z\s'-]", '', words)
        words = words.split()
        # print(words)

        # Fine-grained sentiment - detect repetitions
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                strong_coeff = 2

        neg_lexicon = {'not', 'never', 'no', 'neither'}
        strong_lexicon = {'realli', 'veri', 'much', 'most', 'absolut', 'ever', 'forev', 'so', 'extrem'}
        negation = 1

        sentiment = 0
        for i in range(len(words)):
            if words[i].endswith("n't") or words[i] in neg_lexicon:
                negation = -1
                continue
            if words[i] in strong_lexicon or words[i].endswith("est"):
              strong_coeff = 2
            if words[i].endswith('i'):
                candidate_i = words[i]
                candidate_y = words[i][:-1] + 'y'
                if candidate_i in sentiments:
                    sentiment = sentiments[candidate_i]
                elif candidate_y in sentiments:
                    sentiment = sentiments[candidate_y]
            else:
                candidate = words[i]
                if candidate in sentiments:
                    sentiment = sentiments[candidate]
            if sentiment != 0:
              sentiment *= negation
              negation = 1
            if abs(sentiment) == 2:
                sentiment //= 2
                strong_coeff = 2
        
        return sentiment * strong_coeff
              
    def give_recommendations(self, recommendations, num_recs=3):
        """
        Helper function that was supposed to be called from process()
        Takes in a list movieIDs and compile chatbot's response that contains movie recommendations

        @param recommendations: list of movieIDs to recommend
        @param num_recs: number of movies to recommend during a single turn of the conversation
        @return rec_message: string of complete recommendation message
        @return remaining: list containing the remaining movieIDs to recommend should the user asks for more recommendations
        """
        titles = [self.movie_titles[movieID] for movieID in recommendations]
        patterns = dict()
        patterns[3] = """Given what you told me, I think you would like the following movies: "{}", "{}", and "{}". Would you like more recommendations?"""
        patterns[2] = """Given what you told me, I think you would like the following movies: "{}" and "{}". Would you like more recommendations?"""
        patterns[1] = """Given what you told me, I think you would like the following movie: "{}". Would you like more recommendations?"""
        rec_message = pattern[num_recs].format(*recommendations[:num_recs])
        remaining = recommendations[num_recs:]
        return rec_message, remaining

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie
        """
        sentiments = dict()
        f = open('data/sentiment.txt')
        p = PorterStemmer()
        for line in f:
          kv = line.rstrip().split(',')
          key = p.stem(kv[0], 0, len(kv[0])-1)
          if kv[1] == 'pos':
            sentiments[key] = 1
          else:
            sentiments[key] = -1
        f.close()
        movie_sents = []
        sentences = preprocessed_input.split(".")
        for i in range(len(sentences) - 1):
          movie_sents += self.extract_helper(p, sentiments, sentences[i])
        return movie_sents

    def extract_helper(self, p, sentiments, sentence):
        titles = self.extract_titles(sentence)

        # For lines that do not have punctuations in the end
        if sentence and sentence[-1].isalpha():
          sentence = sentence + '\n'

        # Use Porter Stemmer on the input
        words = ''
        word = ''
        for c in sentence:
          if c.isalpha():
            word += c.lower()
          else:
            if word:
              words += p.stem(word, 0, len(word)-1)
              word = ''
            words += c.lower()

        # Remove the movie titles and puncuation
        just_words = re.sub('"(.*?)"', '', words)
        just_words = re.sub("[^a-zA-Z\s'-]", '', just_words)
        just_words = just_words.split()
        words = words.split()

        neg_lexicon = {'not', 'never', 'no', 'neither'}
        negation = 1
        same_sent = 0
        same_lexicon = {'both', 'and', 'either', 'or', 'along', 'with', 'as', 'well'}
        diff_lexicon = {'but', 'however', 'although', 'while', 'yet', 'though', 'except'}
        for i in range(len(just_words)):
          if just_words[i] in same_lexicon:
            same_sent = 1
          elif just_words[i] in diff_lexicon:
            same_sent = -1

        sentiment = 0
        movie_sentiments = []
        for i in range(len(words)):
          if words[i][0] == '"':
            movie_sentiments.append((titles[0], sentiment))
            if same_sent == 1:
              for j in range(len(titles[1:])):
                movie_sentiments.append((titles[j + 1], sentiment))
            elif same_sent == -1:
              for j in range(len(titles[1:])):
                movie_sentiments.append((titles[j + 1], sentiment * -1))
            return movie_sentiments
          if sentiment != 0:
            if 'but' in words[i] or 'yet' in words[i] or 'still' in words[i]:
              sentiment = 0
            continue
          if words[i].endswith("n't") or words[i] in neg_lexicon:
            negation = -1
            continue
          if words[i].endswith('i'):
            candidate_i = words[i]
            candidate_y = words[i][:-1] + 'y'
            if candidate_i in sentiments:
              sentiment = sentiments[candidate_i]
            elif candidate_y in sentiments:
              sentiment = sentiments[candidate_y]
          else:
            candidate = words[i]
            if candidate in sentiments:
              sentiment = sentiments[candidate]
          if sentiment != 0:
            sentiment *= negation
            negation = 1
        return movie_sentiments

    @staticmethod
    def edit_distance(s, t):
        m = len(s)
        n = len(t)
        d = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1, m + 1):
            d[i][0] = i
        for j in range(1, n + 1):
            d[0][j] = j
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if s[i - 1] == t[j - 1]:
                    substitutionCost = 0
                else:
                    substitutionCost = 2
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + substitutionCost)
        return d[m][n]
        

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """

        title = title.lower()

        # Extract the movie title (target) from input string
        input_extractor = re.compile("(?P<article>(the\s|an\s|a\s)?)(?P<movie>.*(?<!\(\d{4}\)))(?P<year>\(\d{4}\))?$", flags=re.IGNORECASE)
        input_matches = input_extractor.match(title)
        if input_matches == None:
            return []
        target = input_matches.group('movie').strip() if input_matches.group('movie') else ""

        # Compile the regular expression to extract titles from our movie list
        database_extractor = re.compile("(.*?)(,\s(The|An|A))?\s\(\d{4}.*\)", flags=re.IGNORECASE)

        min_distance = float("inf")
        closest_movies = []
        for index, movie in enumerate(self.movie_titles):
            movie_title = database_extractor.match(movie)
            if movie_title == None:
                movie_title = movie
            else:
                movie_title = movie_title.group(1).strip()
            movie_title = movie_title.lower()
            if movie_title.startswith('the'):
                movie_title = movie_title[3:]
            if movie_title.startswith('a'):
                movie_title = movie_title[1:]
            if movie_title.startswith('an'):
                movie_title = movie_title[2:]
            dist = self.edit_distance(movie_title, target)
            if dist <= max_distance:
                if dist < min_distance:
                    min_distance = dist
                    closest_movies = [index]
                elif dist == min_distance:
                    closest_movies.append(index)
        return closest_movies

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """
        nums = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        possible_movies = []
        if clarification == "most recent":
            most_recent_year = float("-inf")
            most_recent_movie = None
            for c in candidates:
                match = re.search("\((?P<year>\d{4})\)",self.movie_titles[c])
                if match != None and int(match.group("year")) > most_recent_year:
                    most_recent_year = int(match.group("year"))
                    most_recent_movie = c
            possible_movies.append(most_recent_movie)
        else:
            for i in range(0,len(nums)):
                if nums[i] in clarification:
                    possible_movies.append(candidates[i])
                    return possible_movies            
            for c in candidates:
                match = re.search("[^\w]" + re.escape(clarification) + "[^\w]", self.movie_titles[c], flags = re.IGNORECASE)
                if match != None:
                    possible_movies.append(c)
            if len(possible_movies) == 0 and clarification.isdigit():
                possible_movies.append(candidates[int(clarification)-1])
                return possible_movies
        if len(possible_movies) == 0:
            closest_movie = None
            smallest_dist = float("inf")
            for c in candidates:
                dist = self.edit_distance(clarification, self.movie_titles[c])
                if dist < smallest_dist:
                    closest_movie = c
                    smallest_dist = dist
            if closest_movie != None:
                possible_movies.append(closest_movie)
        return possible_movies

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        binarized_ratings = np.zeros_like(ratings)

        for movie in range(len(ratings)):
          for user in range(len(ratings[movie])):
            if ratings[movie][user] > threshold:
              binarized_ratings[movie][user] = 1
            elif 0 < ratings[movie][user] <= threshold:
              binarized_ratings[movie][user] = -1
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        similarity = 0
        dot = np.dot(u, v)
        norm_u = LA.norm(u)
        norm_v = LA.norm(v)
        prod = norm_u * norm_v
        if prod == 0:
            return dot
        similarity = dot / prod
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        ratings_map = {}
        ratings = []
        for i in range(ratings_matrix.shape[0]):
          if user_ratings[i] == 0:
            sum = 0
            for j in range(user_ratings.size):
              if user_ratings[j] != 0:
                sim = self.similarity(ratings_matrix[i], ratings_matrix[j])
                score = user_ratings[j]
                sum += sim * score
            ratings_map[sum] = i
            ratings.append(sum)
        ratings = np.sort(ratings)[::-1]  
        for i in range(k): 
          if ratings_map[ratings[i]] not in recommendations:
            recommendations.append(ratings_map[ratings[i]]) 
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return """
        I'm AEHM Movie Bot. I want to see if I can recommend movies that you'd like. You'll start by telling me about movies you've seen, making sure to put the movie names inside double quotes. You can type :quit at any time to exit.
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
