import pyttsx3
name = 'Peter'
engine = pyttsx3.init()
engine.say("Welcome to class, "+name)
engine.runAndWait()