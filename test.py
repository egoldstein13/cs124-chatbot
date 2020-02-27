from chatbot import Chatbot

bot = Chatbot()
line = 'I loved "10 Things I Hate About You"'

# print(bot.extract_sentiment(line))
print(bot.find_movies_closest_to_title('Blargdeblargh', 4))