from chatbot import Chatbot

bot = Chatbot()
line = '"fate" is a worse movie!'

print(bot.extract_sentiment(line))
# print(bot.find_movies_closest_to_title('Blargdeblargh', 4))