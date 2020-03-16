from bot_config import BOT_CONFIG
import nltk

intents = BOT_CONFIG["intents"]
list_of_intensts = []

for keys in intents:
    list_of_intensts.append(keys)

for it in list_of_intensts:
    for other in list_of_intensts:
        distance = nltk.edit_distance(it.lower(), other.lower())
        similarity = 1 - min(1, distance / len(other))
        if similarity > 0.9:
            other_examples = (intents[other])['examples']
            other_responses = (intents[other])['responses']
