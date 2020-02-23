# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import re
import numpy as np
from numpy import linalg as LA
from PorterStemmer import PorterStemmer

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'moviebot'
        
        self.creative = creative
        #self.movies_learned = 0
        # state variables for recommend
        self.time_to_recommend = 0 # the system is ready to recommend
        self.user_wants_recommend = 0 # the user says 'yes'
        self.recommended_movies = []
        self.next_movie_to_recommend = 0
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.movie_titles = [i[0] for i in self.titles] # extract just the titles into a single array
        self.sentiment = movielens.sentiment()
        self.num_ratings = 0
        self.threshold = 0.5

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

        greeting_message = "Hey there!"

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
    
    def handle_recommendation(self, line, response):
        if self.time_to_recommend == 1:     
            if not line.lower() == "yes":
                return "Sorry, I don't understand. Please say yes to continue or :quit to exit."

            if self.user_wants_recommend == 0: # beginning of recommendations
                self.user_wants_recommend = 1
                self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
            #else: # continue to recommend (if there are multiple recommendations)
                    
            if len(self.recommended_movies) == 0:
                self.time_to_recommend = 0
                return "Unfortunately I can't find any movies to recommend just yet. But let's keep going. Tell me about another movie."
            elif self.next_movie_to_recommend < len(self.recommended_movies):
                response = "I recommend \"" + self.recommended_movies[self.next_movie_to_recommend] + "\"."
                self.next_movie_to_recommend = self.next_movie_to_recommend + 1
                
                if self.next_movie_to_recommend == len(self.recommended_movies):
                    response = response +  " Well that's all I have for now! Type :quit to exit."
                else:
                     response = response + " Would you like another recommendation?"
                return response

        else:
            self.time_to_recommend = 1
            return "You've told me your opinions about 5 movies--awesome! Would you like me to recommend some movies? Type yes to continue."
    
    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

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

        if self.creative:
            response = "TODO: responses in creative mode"
        extracted_movies = self.extract_titles(line)
        # I'm not entirely sure which of the code below is just for starter mode and which is for both modes
        if not self.creative:
            if self.time_to_recommend == 1:
                return self.handle_recommendation(line)
           
            if len(extracted_movies) == 0:
                return "Oops! You forgot to put your movie title in quotation marks."
            elif len(extracted_movies) > 1:
                return "You've mentioned more than one movie. Can you please tell me about them one at a time?"
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
                if sentiment == 1:
                    self.num_ratings = self.num_ratings + 1
                    response = "So you liked " + movie + ", huh? " # the extra space here is on purpose because we will add to the response
                elif sentiment == -1:
                    self.num_ratings = self.num_ratings + 1
                    response =  "Sounds like you didn't enjoy " + movie + ". "
                if self.num_ratings >= 5:
                    #self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
                    #self.time_to_recommend = 1
                    return self.handle_recommendation(line, response)
                else:
                    return response + "Tell me about another movie."

        # we shouldn't get here if the code above works, but just in case... 
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

        if not self.creative:
          regex_with_quotes = r'\"(.*?)\"'
          movies = re.findall(regex_with_quotes, preprocessed_input)

        return movies

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
        #print(self.movie_titles)
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

        neg_lexicon = {'not', 'never', 'no', 'neither'}
        negation = 1

        sentiment = 0
        for i in range(len(words)):
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
          sentiment *= negation
        
        return sentiment
              
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
      pattern[3] = """Given what you told me, I think you would like the following movies: "{}", "{}", and "{}". Would you like more recommendations?"""
      pattern[2] = """Given what you told me, I think you would like the following movies: "{}" and "{}". Would you like more recommendations?"""
      pattern[1] = """Given what you told me, I think you would like the following movie: "{}". Would you like more recommendations?"""
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
        pass

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

        pass

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
        pass

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
        if self.num_ratings >= 5:
          for i in range(k): 
            recommendations.append(ratings_map[ratings[i]]) 
        else:
          for i in range(k): 
            if ratings[i] >= self.threshold:
              recommendations.append(ratings_map[ratings[i]]) 
            else:
              return None
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
        I'm MovieBot 1.0. I'd like to see if I can recommend movies that you'd like. But first, tell me what you think about a movie you've seen. Please make sure to put the name of the movie within double quotes. You can type :quit at any time to exit.
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
